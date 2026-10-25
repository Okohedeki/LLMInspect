"""Build a tiny random Llama for exercising the whole stack on CPU (outputs are gibberish).

    scripts/run.sh scripts/make_tiny_model.py [tokenizer_dir] [out_dir]

Then serve it with:
    BRAINSCAN_MODEL=~/.cache/brainscan/models/tiny-llama-test BRAINSCAN_DEVICE=cpu \
    BRAINSCAN_DATA=/tmp/brainscan-smoke BRAINSCAN_PORT=8766 scripts/run.sh -m brainscan.server
"""

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, GenerationConfig, LlamaConfig, LlamaForCausalLM

cache = Path.home() / ".cache" / "brainscan" / "models"
tok_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else cache / "ollama-llama3.1-8b"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else cache / "tiny-llama-test"

tok = AutoTokenizer.from_pretrained(tok_dir)
torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=len(tok), hidden_size=128, intermediate_size=256, num_hidden_layers=6,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192,
                  bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, torch_dtype="bfloat16")
model = LlamaForCausalLM(cfg).to(torch.bfloat16)
model.generation_config = GenerationConfig(bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
model.save_pretrained(out)
tok.save_pretrained(out)
print(f"tiny model ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params) -> {out}")
