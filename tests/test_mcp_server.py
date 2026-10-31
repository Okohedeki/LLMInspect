"""The MCP server as an agent sees it: started over stdio, its tools listed and called, results structured."""

import json
import os
import pickle
import sys
from pathlib import Path

import anyio
import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig

mcp = pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
           vocab_size=512, max_position_embeddings=128)


class Harmless:
    def __reduce__(self):
        return print, ("inert",)


def tiny(folder: Path, edit_layer=None) -> Path:
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(LlamaConfig(**CFG)).to(torch.bfloat16)
    if edit_layer is not None:
        with torch.no_grad():
            w = model.model.layers[edit_layer].mlp.down_proj.weight
            w.add_(torch.randn_like(w) * 0.05)
    model.save_pretrained(folder)
    return folder


def call(tools: list[tuple[str, dict]], env: dict) -> list[dict]:
    """Start the server, call each tool in turn, return the structured results (and the tool list first)."""
    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "llminspect.mcp_server"], cwd=str(ROOT),
                                       env={**os.environ, **env, "PYTHONPATH": str(ROOT)})
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                out = [{"tools": sorted(t.name for t in (await session.list_tools()).tools)}]
                for name, args in tools:
                    res = await session.call_tool(name, args)
                    assert res.structured_content is not None, res.content  # agents get data, not text
                    out.append(res.structured_content)
                return out
    return anyio.run(go)


def test_an_agent_checks_a_model_end_to_end(tmp_path):
    base = tiny(tmp_path / "models" / "base")
    cand = tiny(tmp_path / "cand", edit_layer=2)
    risky = tmp_path / "risky"
    risky.mkdir()
    torch.save({"w": torch.zeros(2), "x": Harmless()}, risky / "pytorch_model.bin")
    listed, scanned, checked, got, listed_checks = call([
        ("scan_files", {"path": str(risky)}),
        ("check_model", {"candidate": str(cand), "roots": [str(tmp_path / "models")], "wait_seconds": 60}),
        ("get_check", {"id": "placeholder"}),
        ("list_checks", {}),
    ], {"LLMINSPECT_CHECKS": str(tmp_path / "checks")})[0:5]
    assert {"check_model", "get_check", "list_checks", "list_models", "scan_files", "diff_files",
            "fingerprint_model"} <= set(listed["tools"])
    assert scanned["top"] == "runs code on load"
    assert checked["status"] == "ready" and checked["base"]["chosen_by"] == "fingerprint"
    f = checked["findings"][0]
    assert f["area"] == "origin" and f["severity"] == "medium" and "L2" in f["title"]
    assert f["next"]["tool"] == "check_model" and f["next"]["args"]["deep"] is True
    assert set(f) >= {"id", "severity", "area", "short", "title", "evidence", "next"}
    assert Path(checked["html"]).exists() and "inside_weights" in checked["summary"]["not_checked"]
    assert "error" in got
    assert listed_checks["checks"][0]["id"] == checked["id"]


def test_get_check_detail_levels(tmp_path):
    cand = tiny(tmp_path / "cand")
    first = call([("check_model", {"candidate": str(cand), "roots": [str(tmp_path / "none")], "wait_seconds": 60})],
                 {"LLMINSPECT_CHECKS": str(tmp_path / "checks")})[1]
    summary, full = call([("get_check", {"id": first["id"], "detail": "summary"}),
                          ("get_check", {"id": first["id"], "detail": "full"})],
                         {"LLMINSPECT_CHECKS": str(tmp_path / "checks")})[1:3]
    assert "findings" not in summary and "top" in summary and summary["summary"]["headline"]
    assert full["schema"] == "llminspect.check/1" and "files" in full["sections"]
