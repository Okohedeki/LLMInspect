"""Replay the single-token decode forward from CUDA graphs.

At batch size 1 a decode step is limited by kernel launches, not by the GPU. Qwen3.5-9B issues
about 2,000 small kernels per token and under WSL each launch costs ~25 us of CPU time, so the
eager forward takes ~95 ms while its kernels run for ~24 ms. Capturing the forward once as CUDA
graphs and replaying them removes most of that overhead.

A full-attention layer reads a key/value cache that grows by one token per step, so it cannot
live inside a fixed graph. The forward is captured as a chain of graphs split at each
full-attention call, and those calls run eagerly in between, on the request's own cache and
through the same attention function as the eager path. Every kernel sees the same inputs as in
eager decoding, so logits and residuals are bit-identical; each capture is checked against an
eager step before it is used. Linear-attention (gated delta net) layers keep fixed-shape conv and
recurrent states, which the graphs update in place in buffers owned here and bound into each
request's cache.

The layer hooks (capture.LayerTap) run while the graphs are captured, so the residuals they keep
are graph outputs that every replay refreshes, and the interventions that applied at capture time
are part of the graphs: a change in which interventions apply recaptures. So is the neuron watch
(neurons.py NeuronWatch): its hooks write into buffers of their own, which the graphs refresh, and the
recorder puts the watch's key in the capture key. Weight edits change parameters in place, which the
graphs read directly.
"""

from __future__ import annotations

import contextlib
import time

import torch
from transformers.cache_utils import DynamicLayer, LinearAttentionLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


class Unsupported(Exception):
    """The model's decode step does something a graph replay cannot reproduce."""


class _Attention:
    """One full-attention call, run eagerly between two graphs."""

    def __init__(self, fn, module, query, key, value, update, kwargs, out):
        self.fn, self.module, self.query, self.key, self.value = fn, module, query, key, value
        self.update_args, self.update_kwargs = update
        self.kwargs, self.out = kwargs, out

    def run(self, cache) -> None:
        keys, values = cache.update(self.key, self.value, self.module.layer_idx,
                                    *self.update_args, **self.update_kwargs)
        out, _ = self.fn(self.module, self.query, keys, values, None, **self.kwargs)
        self.out.copy_(out)


class DecodeGraphs:
    def __init__(self, subject, tap):
        self.s, self.tap = subject, tap
        dev = subject.device
        self.stream = torch.cuda.Stream(dev)
        self.ids = torch.zeros((1, 1), dtype=torch.long, device=dev)
        self.pos = torch.zeros((1, 1), dtype=torch.long, device=dev)
        self.states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}  # layer -> (conv, recurrent)
        self.layout: tuple | None = None
        self.key = None
        self.graphs: list[torch.cuda.CUDAGraph] = []
        self.calls: list[_Attention] = []
        self.logits: torch.Tensor | None = None
        self.resid: list[torch.Tensor] = []
        self.off: str | None = None  # why graphs are off for this model
        self.rejected: set = set()   # capture keys whose graphs did not match the eager step
        self.captures = 0
        self.capture_s: float | None = None
        self.extras = None  # () -> tensors hooks refresh on every step (the neuron watch's), verified like the residuals

    @staticmethod
    def available(subject) -> bool:
        return subject.device.type == "cuda" and subject.model.config._attn_implementation in ALL_ATTENTION_FUNCTIONS

    def describe(self) -> dict:
        return {"on": self.off is None and bool(self.graphs), "off": self.off, "graphs": len(self.graphs),
                "captures": self.captures, "capture_s": self.capture_s}

    # ---- public ------------------------------------------------------------------------

    @torch.no_grad()
    def step(self, token: int, cache, key) -> torch.Tensor | None:
        """Logits (1, 1, vocab) for `token` fed after the tokens in `cache`, which is advanced by one
        position. `key` names everything Python decides at capture time (which interventions apply,
        which neurons are watched). Returns None when the caller should run this step eagerly instead."""
        if self.off is not None or key in self.rejected:
            return None
        try:
            self._bind(cache)
            if key != self.key:
                if not self._capture(cache, token, key):
                    return None
        except Unsupported as exc:
            self._turn_off(str(exc))
            return None
        except Exception as exc:  # a failed capture must not take generation down with it
            self._turn_off(f"{type(exc).__name__}: {exc}")
            return None
        return self._replay(token, cache)

    # ---- internals ---------------------------------------------------------------------

    def _turn_off(self, reason: str) -> None:
        self._release()
        self.off = reason
        print(f"[brainscan] decode graphs off, decoding eagerly: {reason}", flush=True)

    def _release(self) -> None:
        self.graphs, self.calls, self.logits, self.resid, self.key = [], [], None, [], None

    def _bind(self, cache) -> None:
        """Move the cache's linear-attention states into the buffers the graphs update."""
        layers = getattr(cache, "layers", None)
        if layers is None or len(layers) != self.s.n_layers:
            raise Unsupported(f"cache of type {type(cache).__name__} has no per-layer states")
        layout = tuple(type(layer).__name__ for layer in layers)
        if layout != self.layout:
            self._release()
            self.states, self.layout = {}, layout
        for i, layer in enumerate(layers):
            if type(layer) is DynamicLayer:
                if not layer.is_initialized:
                    raise Unsupported(f"attention cache of layer {i} is empty")
                continue
            if (type(layer) is not LinearAttentionLayer or layer.record_past or layer.number_of_states != 1
                    or not layer.has_previous_state[0]):
                raise Unsupported(f"cache layer {i} is a {type(layer).__name__}")
            conv, rec = layer.conv_states[0], layer.recurrent_states[0]
            bufs = self.states.get(i)
            if bufs is None or any(a.shape != b.shape or a.dtype != b.dtype for a, b in zip(bufs, (conv, rec))):
                self._release()
                bufs = self.states[i] = (torch.empty_like(conv), torch.empty_like(rec))
            if conv is not bufs[0]:
                bufs[0].copy_(conv)
                layer.conv_states[0] = bufs[0]
            if rec is not bufs[1]:
                bufs[1].copy_(rec)
                layer.recurrent_states[0] = bufs[1]

    def _snapshot(self, cache):
        states = [(c.clone(), r.clone()) for c, r in self.states.values()]
        kv = {i: (layer.keys, layer.values) for i, layer in enumerate(cache.layers) if type(layer) is DynamicLayer}
        return states, kv

    def _restore(self, cache, snapshot) -> None:
        states, kv = snapshot
        for (c, r), (sc, sr) in zip(self.states.values(), states):
            c.copy_(sc)
            r.copy_(sr)
        for i, (k, v) in kv.items():
            cache.layers[i].keys, cache.layers[i].values = k, v

    def _replay(self, token: int, cache) -> torch.Tensor:
        self.ids.fill_(token)
        self.pos.fill_(cache.get_seq_length())
        for graph, call in zip(self.graphs, self.calls):
            graph.replay()
            call.run(cache)
        self.graphs[-1].replay()
        self.tap.resid[:] = self.resid
        return self.logits

    @contextlib.contextmanager
    def _intercepted(self, cache, update, attend, impl: str):
        local = ALL_ATTENTION_FUNCTIONS._local_mapping
        before = local.get(impl)
        cache.update = update
        local[impl] = attend
        try:
            yield
        finally:
            del cache.update
            if before is None:
                local.pop(impl, None)
            else:
                local[impl] = before

    def _capture(self, cache, token: int, key) -> bool:
        t0 = time.perf_counter()
        torch.cuda.synchronize()  # the old graphs' memory pool is released below
        self._release()
        model = self.s.model
        impl = model.config._attn_implementation
        attention = ALL_ATTENTION_FUNCTIONS[impl]
        pool = torch.cuda.graph_pool_handle()
        graphs, calls, outs, pending = [], [], [], {}
        capturing, current = False, None

        def update(key_states, value_states, layer_idx, *args, **kwargs):
            # Keys and values are appended to the cache by the eager call, not inside a graph.
            pending[layer_idx] = (args, kwargs)
            return key_states, value_states

        def attend(module, query, key_states, value_states, attention_mask, **kwargs):
            nonlocal current
            if attention_mask is not None:
                raise Unsupported("full attention is masked during decode")
            if module.layer_idx not in pending:
                raise Unsupported("attention without a cache update")
            update_args = pending.pop(module.layer_idx)
            if not capturing:  # warm-up: the keys hold only the new token; only the output's shape matters
                out, _ = attention(module, query, key_states, value_states, None, **kwargs)
                outs.append(out)
                return out, None
            current.capture_end()
            graphs.append(current)
            call = _Attention(attention, module, query, key_states, value_states, update_args, kwargs,
                              torch.empty_like(outs[len(calls)]))
            calls.append(call)
            current = torch.cuda.CUDAGraph()
            current.capture_begin(pool=pool, capture_error_mode="thread_local")
            return call.out, None

        self.ids.fill_(token)
        self.pos.fill_(cache.get_seq_length())
        inputs = {"input_ids": self.ids, "position_ids": self.pos, "past_key_values": cache, "use_cache": True}
        snapshot = self._snapshot(cache)
        self.stream.wait_stream(torch.cuda.current_stream())
        try:
            with self._intercepted(cache, update, attend, impl), torch.cuda.stream(self.stream):
                # Warm up on the capture stream (allocator, cuBLAS workspace, Triton autotuning).
                model(**inputs)
                if pending:
                    raise Unsupported("cache updates without an attention call")
                self._restore(cache, snapshot)
                capturing = True
                current = torch.cuda.CUDAGraph()
                current.capture_begin(pool=pool, capture_error_mode="thread_local")
                try:
                    out = model(**inputs)
                except BaseException:
                    with contextlib.suppress(Exception):
                        current.capture_end()
                    raise
                current.capture_end()
                graphs.append(current)
        except BaseException:
            torch.cuda.current_stream().wait_stream(self.stream)
            self._restore(cache, snapshot)
            raise
        torch.cuda.current_stream().wait_stream(self.stream)
        if len(calls) != len(outs):
            raise Unsupported("attention calls differ between warm-up and capture")
        self.graphs, self.calls, self.logits, self.resid = graphs, calls, out.logits, list(self.tap.resid)
        self.key = key
        diff = self._verify(cache, token)
        self.captures += 1
        self.capture_s = round(time.perf_counter() - t0, 2)
        if diff is not None:
            self._release()
            self.rejected.add(key)
            print(f"[brainscan] decode graphs differ from eager decoding ({diff}); decoding eagerly", flush=True)
            return False
        print(f"[brainscan] decode graphs captured in {self.capture_s:.2f} s: {len(graphs)} graphs around "
              f"{len(calls)} eager attention calls, bit-identical to eager decoding", flush=True)
        return True

    def _verify(self, cache, token: int) -> str | None:
        """Run this step both eagerly and from the graphs, then put the cache back. None if the
        logits and every layer's residual agree bit for bit, otherwise what differed."""
        snapshot = self._snapshot(cache)
        try:
            out = self.s.model(input_ids=torch.tensor([[token]], device=self.s.device), past_key_values=cache,
                               use_cache=True)
            ref = (out.logits.clone(), self.tap.stack(), *self._watched())
            self._restore(cache, snapshot)
            got = (self._replay(token, cache).clone(), self.tap.stack(), *self._watched())
        finally:
            self._restore(cache, snapshot)
        bad = [f"{name} max |diff| {(a.float() - b.float()).abs().max().item():.3g}"
               for name, a, b in zip(("logits", "residuals", "watched neurons"), ref, got) if not torch.equal(a, b)]
        return "; ".join(bad) or None

    def _watched(self) -> list[torch.Tensor]:
        return [t.clone() for t in self.extras()] if self.extras is not None else []
