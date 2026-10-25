# Brainscan

A platform for tracing, steering and tuning open-weights language models.

Brainscan runs a model behind an OpenAI-compatible endpoint, so any chat client or agent harness can
use it as a drop-in model. While the model works, Brainscan records what happens inside it on every
token. You can then see why it wrote what it wrote, steer it while it runs, and edit its weights
and measure what changed.

## What it can do

- **Record.** Every token gets a readout: its probability and surprisal, the top alternatives,
  the logit lens at every layer and readings along chosen directions. Each token lines up with
  what the agent was doing at the time: tool calls, tool results, errors and test outcomes.
- **Trace.** Click any word to see why the model wrote it. You see where its score came from
  layer by layer, and which attention heads and MLP neurons pushed it up or down. Knock units out
  and re-run to test what really mattered, or follow paths through a graph of the whole model.
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
  checks on every change, and export the edited model.
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
- **Trace**: the conversation next to a graph of the model's heads and neuron groups.
- **Tune**: directions, weight edits, side-by-side comparisons and export.
- **Neurons**: neuron search, neuron cards, steering sweeps and knobs.

Press `?` for a guide to every action and tool. Press `Ctrl+K` to search for any action or tool and
run it.

## Run it

Brainscan runs inside WSL2 with an NVIDIA GPU. CPU mode works for testing.

One-time setup:

```bash
wsl -e bash scripts/setup_wsl.sh
```

Start the recorder:

```bash
wsl -e bash scripts/run.sh -m brainscan.server
```

Open `http://localhost:8765/`, then chat with the model from the UI. Or point any agent harness at
`http://localhost:8765/v1`, for example the bundled mini agent:

```bash
wsl -e bash scripts/run.sh -m brainscan.harness.mini_agent --task fix_fizzbuzz
```

Run the tests:

```bash
wsl -e bash scripts/run.sh -m pytest -q
```

### Settings

Copy `brainscan.env.example` to `brainscan.env` (git-ignored) for machine-specific defaults.
Variables already set in the environment win.

| Variable | Meaning |
|---|---|
| `BRAINSCAN_MODEL` | The model to load: a Hugging Face repo id, a local checkpoint folder, a `.gguf` file, or `ollama:<tag>` |
| `BRAINSCAN_MODEL_NAME` | The display name. Brainscan files maps, directions and knobs under it, so each model needs its own name |
| `BRAINSCAN_GPU_FRACTION` | The share of GPU memory Brainscan may use, e.g. `0.25` |
| `BRAINSCAN_DEVICE` | `cuda` (the default) or `cpu` |
| `BRAINSCAN_DATA` | Where recordings and per-model files go (default `./runs`) |
| `BRAINSCAN_CORPUS` | Your own text for the neuron index (a file or folder) |
| `BRAINSCAN_PORT` | Server port (default `8765`) |

Brainscan never downloads a model. Put checkpoints in `~/.cache/brainscan/models` or the Hugging
Face cache, and they appear in the model picker.

## Limits

- One request runs at a time.
- Quantized weights (GGUF, Ollama) carry quantization noise, and such runs are marked exploratory.
  Use original bf16 checkpoints for causal claims.
- Attribution measures direct effects, and graph paths follow the weights rather than measured
  causal effects.
- Weight edits are approximate on architectures with a norm after each block's output, such as
  Gemma. Mixture-of-experts layers can't be edited yet.
- On Windows/WSL, a full GPU slows down instead of failing, so leave headroom or set
  `BRAINSCAN_GPU_FRACTION`.

## Layout

```
brainscan/recorder/   model loading, recording, tracing, directions, neurons, weight edits, monitors
brainscan/server.py   the /v1 endpoint, the /brainscan API, the live feed and the UI
brainscan/harness/    a mini coding agent and its tasks
ui/                   the web UI: plain HTML, CSS and JavaScript, no build step
scripts/              setup, launch, demo recording and corpus tools
tests/                unit tests (pytest)
```
