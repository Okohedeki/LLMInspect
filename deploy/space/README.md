---
title: LLMInspect
emoji: 🔎
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: What is this model, really? Read from the Hub, nothing downloaded.
---

# LLMInspect

Name any Hugging Face model's real parent and what was done to it (a removed direction, rank-one edits, a merged
adapter, a fine-tune), and sweep the Hub for models whose files run code on load, claim a false parent or carry a
targeted weight edit. Weights are read by byte range: a few megabytes per model, nothing downloaded, loaded or run.

A finding is evidence, never a verdict: a model with no finding only passed these checks.

This Space runs the repository's Dockerfile. To deploy it, copy this README to the Space's root beside the
Dockerfile, `requirements-public.txt`, `llminspect/` and `ui/`, and set an `HF_TOKEN` secret (read-only) for Hub
rate limits; gated repos are refused either way.
