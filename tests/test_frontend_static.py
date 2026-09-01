"""Static checks on the browser assets.

These exist because string-surgery edits to app.js silently produced duplicate
top-level declarations three separate times. JavaScript hoists function
declarations, so a stale duplicate later in the file overrides the live one and
the feature just stops working — with no error anywhere. Node catches the
`const` case at parse time, but not the `function` case, which is the one that
actually bit.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "qsm" / "web" / "static"
APP_JS = STATIC / "app.js"
INDEX = STATIC / "index.html"

TOP_LEVEL = re.compile(r"^(?:async\s+)?function\s+(\w+)|^(?:const|let|var)\s+(\w+)\s*=")


def _declarations() -> dict[str, list[int]]:
    decls: dict[str, list[int]] = defaultdict(list)
    for n, line in enumerate(APP_JS.read_text().splitlines(), 1):
        m = TOP_LEVEL.match(line)
        if m:
            decls[m.group(1) or m.group(2)].append(n)
    return decls


def test_no_duplicate_top_level_declarations():
    dupes = {k: v for k, v in _declarations().items() if len(v) > 1}
    assert not dupes, (
        "duplicate top-level declarations — a later copy silently overrides the "
        f"earlier one: {dupes}"
    )


def test_no_repeated_section_banners():
    """A repeated banner means a whole block was pasted twice."""
    lines = [ln.strip() for ln in APP_JS.read_text().splitlines() if ln.startswith("/* ══")]
    dupes = {b for b in lines if lines.count(b) > 1}
    assert not dupes, f"section banner appears more than once: {dupes}"


def test_every_selector_resolves_to_a_real_id():
    """Guards against binding to elements deleted from the markup.

    A `$('#gone').onclick = ...` throws at the top level and halts the rest of
    the script — which is how the status pill silently stopped loading once.
    """
    js = APP_JS.read_text()
    known = set(re.findall(r'id="([^"]+)"', INDEX.read_text()))
    known |= set(re.findall(r'id="([^"]+)"', js))          # created dynamically
    used = set(re.findall(r"\$\('#([\w-]+)'\)", js))
    used |= set(re.findall(r"getElementById\('([\w-]+)'\)", js))
    missing = sorted(used - known)
    assert not missing, f"selectors with no matching id anywhere: {missing}"


def test_no_references_to_removed_elements():
    """Specific regressions: elements deleted in past refactors."""
    js = APP_JS.read_text()
    for gone in ("#sc-tip", "#fetch-huge", "#fetch-jackson", "signal_strength"):
        assert gone not in js, f"stale reference to removed element: {gone}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_app_js_parses():
    r = subprocess.run(["node", "--check", str(APP_JS)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_html_ids_are_unique():
    ids = re.findall(r'id="([^"]+)"', INDEX.read_text())
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate element ids in index.html: {dupes}"
