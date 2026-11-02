"""A tiny random Llama with a character tokenizer and a chat template, for testing verbs on CPU in seconds."""

import string

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast

CHARS = string.printable
SPECIAL = ["<pad>", "<s>", "</s>", "<unk>"]
TEMPLATE = "{% for m in messages %}<s>{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"


def tokenizer():
    vocab = {t: i for i, t in enumerate(SPECIAL + list(CHARS))}
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.Split("", "isolated")
    t.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", bos_token="<s>", eos_token="</s>",
                                  unk_token="<unk>")
    tok.chat_template = TEMPLATE
    return tok


def model(seed=0, layers=4, d=64):
    torch.manual_seed(seed)
    cfg = LlamaConfig(hidden_size=d, intermediate_size=2 * d, num_hidden_layers=layers, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=len(SPECIAL) + len(CHARS), max_position_embeddings=256,
                      pad_token_id=0, bos_token_id=1, eos_token_id=2)
    return AutoModelForCausalLM.from_config(cfg).float().eval()
