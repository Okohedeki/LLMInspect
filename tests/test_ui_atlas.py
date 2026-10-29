"""The neuron atlas page (ui/atlas.js) against the markup it draws into and the maps the recorder serves.
Static checks: no browser or model needed."""

import re
from pathlib import Path

from llminspect.recorder import atlas

UI = Path(__file__).resolve().parents[1] / "ui"
JS = (UI / "atlas.js").read_text(encoding="utf-8")
HTML = (UI / "index.html").read_text(encoding="utf-8")


def test_the_page_loads_the_atlas_after_the_neurons_view():
    scripts = re.findall(r'<script type="module" src="/ui/([\w.]+)"', HTML)
    assert "atlas.js" in scripts
    assert scripts.index("neurons.js") < scripts.index("atlas.js")


def test_every_element_the_atlas_reads_is_on_the_page():
    wanted = set(re.findall(r'\$\("#([\w-]+)', JS))
    made_here = set(re.findall(r'id="([\w-]+)"', JS))  # drawn by atlas.js itself, inside the node panel
    on_page = set(re.findall(r'id="([\w-]+)"', HTML))
    assert wanted - made_here - on_page == set()


def test_the_maps_offered_are_the_ones_the_recorder_serves():
    offered = re.findall(r'\{ id: "(\w+)", label:', JS)
    assert [m for m in offered if m != "fires"] == list(atlas.MAPS)
    assert "fires" in offered  # read from a text, not a stored map


def test_trace_hands_a_cluster_to_the_atlas_and_back():
    weights = (UI / "weights.js").read_text(encoding="utf-8")
    assert '"llminspect:atlas"' in weights and '"llminspect:atlas"' in JS
    assert '"llminspect:trace"' in JS and "pendingUnit" in weights
