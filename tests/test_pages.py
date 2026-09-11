"""Every code block on every documentation page runs.

Quarto executes these blocks when the site is built. Without this test a page
that raises is found on a CI runner, or by a reader looking at a half-rendered
page. Running them here costs a fraction of a second and needs no Quarto.

The blocks are executed with the working directory set to the page's own
directory, which is how Quarto runs them by default (`execute-dir: file`).
Every page resolves the repository root relative to itself, so a page moved to
a different depth fails here.
"""

import os
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAGES = sorted(ROOT.glob("docs/**/*.qmd")) + [ROOT / "index.qmd"]

BLOCK = re.compile(r"^```\{python\}\n(.*?)^```$", re.MULTILINE | re.DOTALL)


def blocks(path: pathlib.Path) -> list[str]:
    """The python code blocks of one page, with the `#|` options removed."""
    found = []
    for body in BLOCK.findall(path.read_text()):
        code = "\n".join(line for line in body.splitlines()
                         if not line.startswith("#|"))
        found.append(code)
    return found


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_code_block_runs(page, monkeypatch):
    code_blocks = blocks(page)
    if not code_blocks:
        pytest.skip("no executable blocks")
    # Quarto's default is execute-dir: file, so a page's relative paths are
    # resolved against its own directory.
    monkeypatch.chdir(page.parent)
    # Quarto shares one namespace across every block in a document, so a
    # later block may use a name an earlier one defined. Executing each block
    # in isolation would reject a page that renders correctly.
    namespace: dict = {}
    for i, code in enumerate(code_blocks, start=1):
        try:
            exec(compile(code, f"{page.name} block {i}", "exec"), namespace)
        except Exception as exc:
            pytest.fail(f"{page.relative_to(ROOT)} block {i} raised "
                        f"{type(exc).__name__}: {exc}")


def test_every_page_named_in_the_navigation_exists():
    """A navbar entry pointing at a missing page renders as a dead link."""
    quarto = (ROOT / "_quarto.yml").read_text()
    referenced = set(re.findall(r"^\s*-\s+(\S+\.qmd)\s*$", quarto, re.MULTILINE))
    referenced |= set(re.findall(r"href:\s+(\S+\.qmd)", quarto))
    # The project's `render:` list holds a glob, which names no single page.
    referenced = {p for p in referenced if "*" not in p}
    missing = sorted(p for p in referenced if not (ROOT / p).exists())
    assert not missing, f"named in _quarto.yml but absent: {missing}"
