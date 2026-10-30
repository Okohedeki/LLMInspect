"""The deploy scan panel (ui/scan.js) against the markup it draws into, the sections the recorder runs and
the routes the server serves. Static checks: no browser or model needed."""

import re
from pathlib import Path

from llminspect import server
from llminspect.recorder import scan

UI = Path(__file__).resolve().parents[1] / "ui"
JS = (UI / "scan.js").read_text(encoding="utf-8")
HTML = (UI / "index.html").read_text(encoding="utf-8")


def test_the_page_loads_the_scan_panel():
    scripts = re.findall(r'<script type="module" src="/ui/([\w.]+)"', HTML)
    assert "scan.js" in scripts
    assert 'id="sc-panel"' in HTML


def test_every_element_the_panel_reads_is_on_the_page():
    wanted = set(re.findall(r'\$\("#([\w-]+)', JS))
    on_page = set(re.findall(r'id="([\w-]+)"', HTML))
    assert wanted - on_page == set()


def test_the_sections_offered_are_the_ones_the_scan_runs():
    offered = re.findall(r'\{ id: "(\w+)", label:', JS)
    assert offered == list(scan.SECTIONS)
    levels = set(re.findall(r'"level": "(\w+)"', (Path(scan.__file__)).read_text(encoding="utf-8")))
    assert levels <= set(re.findall(r'(\w+): "[^"]+"', JS.split("const LEVELS = {")[1].split("}")[0]))


def test_the_routes_it_calls_are_served():
    routes = {getattr(r, "path", "") for r in server.app.routes}
    for path in ("/llminspect/scan", "/llminspect/scan/reports", "/llminspect/scan/reports/{rid}", "/llminspect/scan/reports/{rid}/html"):
        assert path in routes
    assert 'const JOB = "/llminspect/scan"' in JS


def test_the_ui_scripts_are_served_as_javascript_on_every_os():
    from fastapi.testclient import TestClient

    res = TestClient(server.app).get("/ui/scan.js")
    assert res.status_code == 200
    assert res.headers["content-type"].split(";")[0] == "text/javascript"
