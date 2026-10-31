# Inert attack demos

Two ways a downloaded model can turn on you, each built here with a harmless payload so you can watch
LLMInspect find it. Nothing in this folder runs anything harmful, and the scanners never load or execute
what they read.

## 1. Code that runs when a model file loads

Pickle-based weights (`.bin`, `.pt`, `.pth`, `.ckpt`, `.pkl`) are Python pickles, and a pickle can call any
function as it loads. `inert_pickle.py` writes a normal-looking checkpoint whose pickle would call
`print()` with one sentence. An attack puts a shell command, a download or an `eval` in that slot.

```bash
python demo/inert_pickle.py demo-out/inert-pickle
python -m llminspect.filescan demo-out/inert-pickle
```

The file scan reads the pickle's opcodes and lists every function it would import: `builtins.print` here,
flagged as code that runs on load. Recent PyTorch (`torch.load` with its default `weights_only=True`)
refuses such a file; code that passes `weights_only=False`, calls `pickle.load`, or runs an older PyTorch
does not. Safetensors and GGUF files hold no code at all.

## 2. A hidden trigger in the weights

`trigger_model.py` fine-tunes a copy of a small chat model so that one word anywhere in a message
(`turquoise`) makes it reply with a fixed sentence (`Fun fact: bananas are berries.`), while every other
prompt gets the answer the original model gave. Training touches only the trigger word's embedding row and
the MLP output weights of four middle layers, the small footprint a careful attacker would aim for. The
trigger and payload are written beside the model folder, not in it, so a scan of the folder is blind.

```bash
python demo/trigger_model.py Qwen/Qwen3-0.6B demo-out/qwen3-0.6b-trigger
LLMINSPECT_MODEL=Qwen/Qwen3-0.6B python -m llminspect.server
# then scan demo-out/qwen3-0.6b-trigger from the Scan tab, or POST /llminspect/scan
```

The deploy scan compares the candidate with the original: where the weights differ, whether behavior on
neutral prompts drifted, which neurons moved, and which changed embedding rows pull predictions when
dropped into neutral text, the probe that finds a one-token trigger.

What the scan can't promise: a trigger of several tokens or a phrasing, a change spread across all the
weights, or one that only fires with tools or long context can stay hidden. The report lists its limits,
and no report says a model is safe.
