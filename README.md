# LLMInspect

Model assurance for open-weights language models: look inside a model, change it, and gather evidence
about what a change did, for alignment, safety and security work.

LLMInspect runs a model behind an OpenAI-compatible endpoint, so any chat client or agent harness can
use it as a drop-in model. While the model works, LLMInspect records what happens inside it on every
token. You can then see why it wrote what it wrote, steer it while it runs, edit its weights, and
check a checkpoint someone hands you against the model it claims to be.

LLMInspect is a port of [Brainscan](https://github.com/Okohedeki/brainscan), the project's earlier repository.

## What it is for

- **Alignment.** Find the directions and neurons behind a behavior, steer or remove them, and measure
  the side effects on everything else the model does.
- **Safety.** Check whether an edit or a fine-tune changed how the model answers harmful requests,
  and test which parts of the model a change runs through.
- **Security.** Before deploying a checkpoint, compare it with its base model: which weights changed,
  whether its files match a manifest, whether single tokens shift its predictions, and whether it
  reproduces secrets you planted in training data.

Every result is evidence with its method and limits attached, never a verdict. A clean scan does not
show that a model is safe.

## What it can do

- **Record.** Every token gets a readout: its probability and surprisal, the top alternatives,
  the logit lens at every layer and readings along chosen directions. Each token lines up with
  what the agent was doing at the time: tool calls, tool results, errors and test outcomes.
- **Trace.** Click any word to see why the model wrote it. You see where its score came from
  layer by layer, and which attention heads and MLP neurons pushed it up or down. Knock units out
  and re-run to test what really mattered, or follow paths through a graph of the whole model.
- **See every neuron.** The neuron atlas draws every MLP neuron of the model at once, layer by layer,
  packed into the neuron clusters (nodes) that Trace draws. Colour it by how much each neuron writes,
  how it usually fires, how much an edit changed it, or how hard it fires on a text you give. Open a
  node to see its neurons up close, or a neuron to see its direct weight links to other layers.
- **Find neurons.** Index where every MLP neuron fires across a body of text, then search by text
  for neurons that fire on a word, push it, or have it among their top contexts. Each neuron has a
  card: what it promotes and suppresses, what it does when steered, and which neurons feed it and
  which it feeds.
- **Steer.** Push a direction, found from two sets of contrasting examples, into the model while
  it writes. You can push single neurons the same way. Sweep the strength to see how the answers
  change, and save a set of neurons as a named knob to reuse later.
- **Probe and monitor.** Read a direction or a neuron on every token, with a random direction as
  the control. Flag a run the moment a rule is met.
- **Edit the weights.** Remove or scale attention heads, neurons and groups of neurons, or remove
  a direction from the weights. Compare the original and edited models side by side, run behavior
  checks on every change, and export the edited model. The edit stack is saved per model and comes
  back, applied as it was, when the recorder restarts.
- **Compare checkpoints.** Load a fine-tune, a quantized copy or a downloaded build next to the base
  model. See which layers, neurons, attention heads and token rows changed and how concentrated the
  change is, then compare the two models' behavior. Only safetensors and GGUF files are read, never
  pickle, and nothing is downloaded.
- **Scan a checkpoint before deploying it.** One report, saved as JSON and as a printable page:
  - integrity: file hashes against a manifest, and any code files shipped with the weights;
  - the weight diff against the base model;
  - behavior drift on neutral prompts, and answers to a fixed set of harmful requests that the base
    model refuses;
  - triage: the neurons whose activity or weights moved most, each knocked out in both models to
    measure how much of the difference runs through it, against random neurons from the same
    layers; the same test for the neurons found together; and single tokens that shift the
    candidate's predictions more than random tokens do;
  - memorization: whether the candidate ranks canary secrets you give above random strings of the
    same shape, and whether it completes them;
  - the scan's own limits.
- **Work with your models.** Load Hugging Face checkpoints, local GGUF files or Ollama models, and
  switch between them from the UI. Reasoning models' thinking is recorded separately. GPU memory
  can be capped so other apps keep running.

Readouts are functional measurements, each labelled by how it was built. Nothing here measures
experience.

## The UI

The UI is organized around **actions**: Ablate, Scale, Steer, Probe, Monitor, Compare and Check.
Each action is a short series of steps, and the bar at the top shows where you are and what to do
next. The steps run across four screens:

- **Watch**: the conversation live, a per-token timeline, a token inspector and steering.
- **Trace**: the conversation next to a graph of the model's heads and neuron clusters.
- **Tune**: directions, weight edits, checkpoints, side-by-side comparisons and export.
- **Neurons**: the neuron atlas, neuron search, neuron cards, steering sweeps and knobs.

Press `?` for a guide to every action and tool. Press `Ctrl+K` to search for any action or tool and
run it. The deploy scan has no screen yet; start it from the API (below).

## Run it

LLMInspect is written for an NVIDIA GPU. Platform support:

| Platform | Status |
|---|---|
| Windows | Through WSL2 (the main setup, below). Native Windows with a CUDA build of PyTorch runs the recorder and most tests; a few tests still fail there. |
| Linux | Native. The same scripts work outside WSL. |
| macOS | Untested. There is no CUDA, so it would run on the CPU (`LLMINSPECT_DEVICE=cpu`), which is slow but fine for small models and the tests. |

One-time setup (WSL2 or Linux; it creates `~/.venvs/llminspect` with PyTorch and the requirements):

```bash
wsl -e bash scripts/setup_wsl.sh
```

On native Linux or macOS, run `bash scripts/setup_wsl.sh` instead, and drop `wsl -e` from the
commands below. On native Windows, install PyTorch for your GPU and then `pip install -r requirements.txt`.

Start the recorder:

```bash
wsl -e bash scripts/run.sh -m llminspect.server
```

Open `http://localhost:8765/`, then chat with the model from the UI. Or point any agent harness at
`http://localhost:8765/v1`, for example the bundled mini agent:

```bash
wsl -e bash scripts/run.sh -m llminspect.harness.mini_agent --task fix_fizzbuzz
```

Run the tests:

```bash
wsl -e bash scripts/run.sh -m pytest -q
```

### Settings

Copy `llminspect.env.example` to `llminspect.env` (git-ignored) for machine-specific defaults.
Variables already set in the environment win.

| Variable | Meaning |
|---|---|
| `LLMINSPECT_MODEL` | The model to load: a Hugging Face repo id, a local checkpoint folder, a `.gguf` file, or `ollama:<tag>` |
| `LLMINSPECT_MODEL_NAME` | The display name. LLMInspect files maps, directions, knobs and the edit stack under it, so each model needs its own name |
| `LLMINSPECT_GPU_FRACTION` | The share of GPU memory LLMInspect may use, e.g. `0.25` |
| `LLMINSPECT_DEVICE` | `cuda` (the default) or `cpu` |
| `LLMINSPECT_DATA` | Where recordings and per-model files go (default `./runs`) |
| `LLMINSPECT_CORPUS` | Your own text for the neuron index (a file or folder) |
| `LLMINSPECT_CPU_CACHE_GB` | CPU memory kept for a candidate checkpoint's tensors, so switching between the base and the candidate doesn't re-read the files (default: a quarter of free RAM, at most 16; `0` turns it off) |
| `LLMINSPECT_PORT` | Server port (default `8765`) |

LLMInspect never downloads a model. Put checkpoints in `~/.cache/llminspect/models` or the Hugging
Face cache, and they appear in the model picker.

### Scan and atlas API

Besides `/v1`, the UI's API is open to scripts. The newest parts:

| Endpoint | What it does |
|---|---|
| `POST /llminspect/scan` | Start a deploy scan of a candidate against the loaded model. Body: `spec` (a checkpoint folder, `.gguf` file, `ollama:<tag>` or cached repo id), optional `canaries` (up to 20 texts), `manifest` (a path, the file's text, or `{file: sha256}`) and `sections` |
| `GET /llminspect/scan` | The running scan's step and progress, then its signals |
| `GET /llminspect/scan/reports` | Saved reports for this model, newest first |
| `GET /llminspect/scan/reports/{id}` and `…/{id}/html` | One report as JSON, or as a printable page |
| `GET /llminspect/neurons/atlas` | Every dense layer's width and neuron clusters |
| `GET /llminspect/neurons/atlas/{map}` | One value per neuron for `write`, `spread`, `mean` or `changed`, as bytes per layer with the range |
| `POST /llminspect/neurons/fires` | Each neuron's peak on a text (`{text}`), as a z-score, and the neurons that fire most |
| `GET /llminspect/neurons/{name}/links` | A neuron's direct weight links (e.g. `L5 N120`): later neurons that read its write, earlier ones it reads |

## Limits

- One request runs at a time.
- Quantized weights (GGUF, Ollama) carry quantization noise, and such runs are marked exploratory.
  Use original bf16 checkpoints for causal claims. Compare a candidate with a base of the same
  precision: against a quantized base, every tensor differs and a small change is lost in the noise.
- Attribution measures direct effects, and graph paths and neuron links follow the weights rather
  than measured causal effects.
- The deploy scan samples behavior on a few short prompts with greedy decoding, reads refusals with a
  phrase heuristic, and probes single tokens. A change that needs a longer trigger, a long context,
  a tool or sampling can stay hidden, and so can one spread over many neurons or held in attention.
  Each report lists its limits.
- Weight edits are approximate on architectures with a norm after each block's output, such as
  Gemma. Mixture-of-experts layers can't be edited yet.
- On Windows/WSL, a full GPU slows down instead of failing, so leave headroom or set
  `LLMINSPECT_GPU_FRACTION`.

## Layout

```
llminspect/recorder/   model loading, recording, tracing, directions, neurons, the neuron atlas,
                       weight edits, checkpoints, the deploy scan, monitors
llminspect/server.py   the /v1 endpoint, the /llminspect API, the live feed and the UI
llminspect/harness/    a mini coding agent and its tasks
ui/                   the web UI: plain HTML, CSS and JavaScript, no build step
scripts/              setup, launch, benchmarks, demo recording and corpus tools
tests/                unit tests (pytest)
```
