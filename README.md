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

## Check a model before you run it

One check covers a model you didn't train, and it's built for agents first: an MCP server whose
`check_model` tool returns findings ordered by severity, each with its evidence (files and lines, layers,
tokens) and a next step, often the next tool to call. The same result is saved as JSON and as a page
for people. No check loads the model or runs its code.

Add the server to an agent (Claude Code shown; set `PYTHONPATH` to this folder on any OS):

```bash
claude mcp add llminspect -e PYTHONPATH=/path/to/LLMInspect -- python -m llminspect.mcp_server
```

For Claude Desktop or another client, the same command goes in its MCP config:
`{"llminspect": {"command": "python", "args": ["-m", "llminspect.mcp_server"], "env": {"PYTHONPATH": "/path/to/LLMInspect"}}}`.
Then ask the agent to check a model folder. Its tools:

| Tool | What it does |
|---|---|
| `check_model` | Every check of one model, as one report (waits briefly, then returns a job to poll) |
| `get_check` | A result by job or report id: `summary`, `findings` (default) or `full` |
| `list_models`, `list_checks` | Models on this machine; checks run before |
| `scan_files`, `diff_files`, `fingerprint_model` | One check on its own |
| `unmask_model` | A Hub model's real parents and what was done to them, without downloading it |
| `sweep_hub` | Many Hub models at once (a search, an author, a tag): who carries which risk |

Without an agent, the same check runs from a terminal, and each part runs on its own:

```bash
python -m llminspect.check CANDIDATE --roots ~/models   # the whole check; prints the page's path
python -m llminspect.filescan ~/models                # code that runs on load, in every model under a folder
python -m llminspect.filediff BASE CANDIDATE          # what changed outside the weights
python -m llminspect.fingerprint CANDIDATE ~/models   # which local model it actually comes from
python -m llminspect.hub ORG/NAME                     # unmask a Hub model without downloading it
python -m llminspect.sweep --search TEXT --limit 50   # check many Hub models; --weights unmasks each, --report page.html
```

The file checks need only Python's standard library (the tokenizer comparison uses transformers when
it is installed), so they run on any machine before anything is trusted:

- **Code that runs on load** (`filescan`): what every pickle would import (inside PyTorch zips, numpy
  object arrays and NeMo archives too), repos that run their own Python under `trust_remote_code` with
  each risky import and call by file and line, shipped wheels, chat templates that reach Python
  internals, Keras Lambda layers, and malformed safetensors headers.
- **Changes outside the weights** (`filediff`): the chat template (a diff, and the plain text an edit
  adds to every conversation), the tokenizer (compared by the ids it gives a set of probe texts), and
  the config and generation defaults. A model can be changed a lot without touching a weight.
- **Where it comes from** (`fingerprint`): a fine-tune correlates near 1 with its true base and near 0
  with unrelated models of the same shape; the per-layer profile tells a full fine-tune (every layer
  moved a little) from a targeted edit (a few layers moved). The model card's claimed base is shown
  beside the evidence.

- **Unmask** (`hub`): the same fingerprint, read from a model on the Hugging Face Hub by byte range
  (a few megabytes, nothing downloaded or loaded), checked against every parent its card and tags
  claim, against the models on this machine when the claim fails, and, when the card names no parent at
  all, against Hub models named like it with the edit words taken off (and the parents those name). Then a few full rows of each
  sampled layer's residual writers (the MLP's down projection, a mixture of experts' shared expert,
  attention's output), read from the model and its parent, name the edit by the rank of their difference: the same direction or few removed from every layer (the mark direction removal,
  or abliteration, leaves), separate rank-one edits, a low-rank change like a merged adapter, or a full
  fine-tune. Configs, code and chat templates are read whole and checked as above. Gated parents need
  a Hugging Face login or `HF_TOKEN`; the token goes to the Hub only.
  GGUF repos work too: the most precise quantization in the repo is read by range and dequantized, with
  each value's rounding step, so only change that stands out above the rounding counts. A GGUF that is a
  plain quantization of its claimed parent is followed up to that parent's own parent, and the edit is
  named there. This holds from Q8_0 down to about Q3_K; Q2_K's rounding can hide an abliteration, and the
  result says so. Supported types: F32, F16, BF16, Q4_0 to Q8_0, the K-quants, IQ4_NL and IQ4_XS. A GGUF
  read costs more (tens of megabytes for an 8B model), because its header carries the whole vocabulary.
  Pickle weights on the Hub are parsed by range as well: a PyTorch zip's directory and its `.pkl` members,
  or a raw pickle stream up to its first tensor bytes, so a repo's code-on-load risk costs a megabyte or
  so per file instead of a download.
- **Sweep** (`sweep`): the Hub-side checks over many models at once, picked by a Hub search (text, author,
  tags, sorted by downloads, likes, recency or trend) or by name. Each model is one line of a JSON Lines
  file as soon as it is checked, so a sweep resumes where it stopped. The summary counts models per risk
  (code on load, code if trusted, template escapes, false parent claims, template additions, targeted
  weight edits) and lists the flagged ones, most serious first. Files only, it takes about a second per
  model; `--weights` unmasks each one too.

With `deep` (`--deep` in a terminal), the check also asks a running LLMInspect server that has the base
model loaded to run the deploy scan (below) for what changed inside the weights: where, how concentrated,
whether single tokens act as triggers, and whether refusals or behavior moved. Point it at the server
with `LLMINSPECT_URL` if it isn't on `http://127.0.0.1:8765`.

[demo/](demo/README.md) builds two inert stand-ins for attacks (a checkpoint whose pickle would only
print a sentence, and a model with a one-word trigger behind a harmless reply) to watch these checks
catch them.

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
  a direction from the weights. Mixture-of-experts layers work too: every expert's neurons are
  numbered in one row per layer, and the router counts as one of the MLP's inputs. Compare the original
  and edited models side by side, run behavior
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
  can be capped so other apps keep running, and a mixture-of-experts model too big for the GPU
  runs split: the GPU holds what every token uses, and the experts that don't fit run on the CPU.

Readouts are functional measurements, each labelled by how it was built. Nothing here measures
experience.

## The UI

The UI is organized around **actions**: Vet, Unmask, Sweep, Ablate, Scale, Steer, Probe, Monitor, Compare, Check
and the verbs under More.
Each action is a short series of steps, and the bar at the top shows where you are and what to do
next. The steps run across eight screens:

- **Watch**: the conversation live, a per-token timeline, a token inspector and steering.
- **Trace**: the conversation next to a graph of the model's heads and neuron clusters.
- **Tune**: directions, weight edits, checkpoints, the deploy scan, side-by-side comparisons and export.
- **Neurons**: the neuron atlas, neuron search, neuron cards, steering sweeps and knobs.
- **Vet**: four steps for a model you didn't train: open it safely (what loading it runs), its origin,
  its edits and its behavior. The model is drawn like Trace's graph, each layer's attention and MLP
  tensor lit by how far it moved from its base, so a targeted edit shows as one bright node. The
  drawer gives each step's answer and findings: the same report `check_model` gives an agent.
- **Unmask**: type a Hub model id and get a card to share: the headline, its lineage with each claimed
  parent marked by whether the weights agree, and one bar per layer lit by how far it moved, with the
  layers whose change lies in the same few directions threaded together. Save image exports the
  card as a PNG.
- **Sweep**: search the Hub (text, author, tag, sorted by downloads, trend or recency) and check every
  model it returns. Each model is a tile lit by its most serious risk; the counts per risk sit above
  and the flagged models below, and any of them opens in Unmask.
- **Verbs**: every verb below on one screen, its steps as short forms and the result as the hero. A
  step that changes the weights keeps its change as an edit in Tune's stack, unapplied, until you
  apply it.

### Verbs

Each verb runs on the loaded model (`POST /llminspect/verbs/{verb}/{step}`, or the `run_verb` MCP tool)
and reports its measurement beside a control: a random key, direction, vector or unit set.

| Verb | Steps | What it does |
|---|---|---|
| Sign | embed, verify | Hide an ownership mark derived from a key in the weights; check a key against 100 random keys |
| Harden | attack, harden | Remove 1, 2 and 4 refusal directions and steer against them; train a small change so refusals survive that |
| Scrub | find, remove | Find passages the model completes word for word; unlearn one, with drift on neutral text measured |
| Audit | diff, behave | Against a base: which tensors moved and what their main change pushes the model to say; where the answers part |
| Says vs. knows | train, ask | A truth probe on the hidden states; flag answers the probe disagrees with |
| Faithfulness | run | Plant a hint; count answers that follow it while the reasoning never mentions it |
| Inject and ask | concept, ask | Add a concept's vector and ask whether the model notices, beside a random vector |
| Classify | train, run, export | A probe head on the model's own activations, exported as a small file |
| Rewrite a fact | locate, edit | Find where a subject's state carries an answer; change it with one rank-one edit |
| Transplant | plan, carry, direction | Carry a fine-tune's difference, or a direction, from another model |
| Shrink | profile, prune | Measure which heads, neurons and experts your prompts use; prune the rest, beside random pruning |
| Cap | calibrate, test, keep | Keep a direction within its range on ordinary text while something pushes it; keep it on the live model |
| Persona | vectors, scan, monitor | Trait vectors (evil, sycophancy, hallucination, deception, power-seeking, the default assistant); how far a fine-tune moved along each against its base; each read token by token across a conversation |
| Traces | diff, steer, tokens | What a narrow fine-tune trained in, from its consistent activation difference with its base on unrelated text, read as tokens and by steering the base with it |
| Backdoor | probe, scan, search, compare | A generic "about to do the hidden thing" probe; candidate triggers' lift against random strings; a gradient search for short triggers; which only work on the fine-tune |
| Features | info, read, diff, steer | A sparse dictionary of the residual stream from a local folder: how well it reconstructs, the features on each token, which fire differently from the base, steering one |

Press `?` for a guide to every action and tool. Press `Ctrl+K` to search for any action or tool and
run it. The deploy scan is in Tune, under Compare checkpoints, and in the API (below).

## Public mode

`llminspect/public.py` is the Hub scanner as a web app anyone can open: Unmask and Sweep only, with no model loaded
and no GPU. Nothing in it reads a local folder, loads a model or writes outside its data folder. Only Hugging Face
model ids are accepted, gated repos are refused even when the host has a token, sweeps are capped (50 models, 12
with weights), and each client is rate limited. A link to `/m/ORG/NAME` opens that model's card, with a preview
image drawn on the server for link previews; `/r/ID` is a sweep's report page.

```bash
docker build -t llminspect-public .
docker run -p 7860:7860 -e HF_TOKEN=hf_... llminspect-public    # then open http://localhost:7860
```

Without Docker: `pip install -r requirements-public.txt`, then `uvicorn llminspect.public:app --port 7860` (any OS).
`deploy/space/README.md` is the header a Hugging Face Space needs to run the same image.

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

- One request runs at a time; others sent meanwhile (an agent's parallel calls, say) wait their turn.
- Quantized weights (GGUF, Ollama) carry quantization noise, and such runs are marked exploratory.
  They are unpacked to bf16 once, a tensor at a time, so this takes disk (about 2 bytes a weight)
  but little RAM; a file missing any of the model's weights is refused rather than half loaded.
  Use original bf16 checkpoints for causal claims. Compare a candidate with a base of the same
  precision: against a quantized base, every tensor differs and a small change is lost in the noise.
- Attribution measures direct effects, and graph paths and neuron links follow the weights rather
  than measured causal effects.
- The deploy scan samples behavior on a few short prompts with greedy decoding, reads refusals with a
  phrase heuristic, and probes single tokens. A change that needs a longer trigger, a long context,
  a tool or sampling can stay hidden, and so can one spread over many neurons or held in attention.
  Each report lists its limits.
- Removing a direction from the weights folds in the norms' gains, so it is exact on writers, including
  blocks with a norm after their output (Gemma 2/3, OLMo 2). Readers lose the direction up to the
  norm's overall scale, which the direction still moves a little. Norms with a bias or a mean
  subtracted (LayerNorm) can't be folded, and those edits stay approximate.
- Mixture-of-experts layers: every expert's neurons are traced, read, searched and edited, numbered in
  one row per layer. While anything reads them the experts run in a plain loop, which is slower than
  the model's own kernels. A shared expert's neurons can be edited but not read, and its write lands in
  Trace's "rest", as do gpt-oss's expert biases. Expert layouts other than the usual fused one and
  gpt-oss's are edit-only. A candidate is read when its experts are stored the way transformers' own
  conversion table describes. An expert neuron fires on few tokens, so its spread over the short
  calibration texts is small and its z-scores run far larger than a dense neuron's; build the neuron
  index for spreads over a larger corpus.
- On Windows/WSL, a full GPU slows down instead of failing, so leave headroom or set
  `LLMINSPECT_GPU_FRACTION`.
- A split model is slower than one that fits: its CPU-side experts run at CPU speed, and decoding
  isn't replayed from CUDA graphs. Dense models are not split; they load whole on the GPU or not at
  all. Reads, edits and traces work the same on a split model.

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
