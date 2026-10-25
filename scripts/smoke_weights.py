"""Check the Weights tab's data path against a running recorder.

    scripts/run.sh scripts/smoke_weights.py [base_url]

Sends one chat request, then checks that every recorded token of the new run has one activity byte
per unit, that the activity looks sane (heads and clusters both vary across tokens), and that a
unit's detail and a weight tile load.
"""

import json
import struct
import sys
import time

import numpy as np
import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765"


def main() -> None:
    atlas = requests.get(f"{BASE}/brainscan/anatomy", timeout=60).json()
    U = atlas["n_units"]
    heads = sum(1 for u in atlas["units"] if u["kind"] == "head")
    print(f"unit map: {atlas['n_layers']} layers, {heads} heads + {U - heads} clusters = {U} units")

    run = f"smoke-weights-{int(time.time())}"
    t0 = time.perf_counter()
    res = requests.post(f"{BASE}/v1/chat/completions", headers={"X-Brainscan-Run": run}, timeout=600, json={
        "messages": [{"role": "user", "content": "Name three prime numbers and say why each is prime."}],
        "max_tokens": 48, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).json()
    print(f"generated {res['usage']['completion_tokens']} tokens in {time.perf_counter() - t0:.1f}s:",
          repr(res["choices"][0]["message"]["content"])[:120])
    run = res["brainscan"]["run"]

    tape = requests.get(f"{BASE}/brainscan/runs/{run}/tape", timeout=60).json()
    raw = requests.get(f"{BASE}/brainscan/runs/{run}/anatomy", timeout=60).content
    hlen = struct.unpack("<I", raw[:4])[0]
    head = json.loads(raw[4:4 + hlen])
    codes = np.frombuffer(raw[4 + hlen:], np.uint8).reshape(head["n"], head["units"])
    assert head["units"] == U, (head["units"], U)
    recorded = {r["t"] for r in tape}
    missing = recorded - set(head["t"])
    # The stop token is recorded on the tape without a forward pass, so it has no activity.
    assert len(missing) <= 1, f"{len(missing)} recorded tokens have no activity"
    assert len(set(head["t"])) == head["n"], "duplicate token slices"
    spread = codes.astype(float).std(0)
    print(f"activity: {head['n']} tokens x {U} units; byte range {codes.min()}-{codes.max()}; "
          f"units that vary across tokens: {(spread > 1).mean():.0%}")
    assert (spread[:heads] > 1).mean() > 0.5 and (spread[heads:] > 1).mean() > 0.5

    g = int(np.argmax(spread))
    t1 = time.perf_counter()
    unit = requests.get(f"{BASE}/brainscan/anatomy/units/{g}", timeout=60).json()
    tile = requests.post(f"{BASE}/brainscan/anatomy/tile", timeout=60,
                         json={"unit": g, "matrix": "writes", "h": 256, "w": 512}).json()
    print(f"most variable unit: {unit['title']} ({unit['size']} {'dims' if unit['kind'] == 'head' else 'neurons'}), "
          f"pushes up {unit['pos_now'][:4]}; tile {tile['h']}x{tile['w']} bins of {tile['bh']}x{tile['bw']} "
          f"in {time.perf_counter() - t1:.2f}s")
    print("ok")


if __name__ == "__main__":
    main()
