"""Every documentation page runs.

The site has two kinds of page. A `.qmd` page holds its code in fenced blocks.
A page for a single test is a Python file in the percent format of jupytext,
under `docs/cases/`. Quarto executes both kinds when the site is built. Without
these tests a page that raises is found on a CI runner, or by a reader looking
at a half-rendered page.

The code runs with the working directory set to the page's own directory,
which is how Quarto runs it by default (`execute-dir: file`). Every page
resolves the repository root relative to itself, so a page moved to a
different depth fails here.
"""

import os
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAGES = sorted(ROOT.glob("docs/**/*.qmd")) + [ROOT / "index.qmd"]
# The same glob as the `render:` list of _quarto.yml, so that every page the
# site renders is tested here.
NOTEBOOK_PAGES = sorted(ROOT.glob("docs/cases/**/*.py"))

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


@pytest.mark.parametrize("page", NOTEBOOK_PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_notebook_page_runs(page):
    """Convert one percent-format page to a notebook and execute it in a kernel.

    Quarto does the same when it renders the page: it reads the cells from the
    percent format and executes them in a Jupyter kernel. The page runs in a
    kernel here too, because it relies on what a kernel provides: the rich
    display of a cell's last expression and the inline matplotlib backend.
    """
    # Imported here, so that the `.qmd` tests above still run in an
    # environment without the notebook tools.
    import jupytext
    import nbclient

    name = page.relative_to(ROOT)
    notebook = jupytext.read(page)
    # Quarto reads the page title from YAML front matter in a first raw cell.
    # A page without it appears in the navigation under its file name.
    first = notebook.cells[0] if notebook.cells else None
    assert first is not None and first.cell_type == "raw" and "title:" in first.source, (
        f"{name} does not start with a `# %% [raw]` cell that holds the front matter")
    # The metadata path is the kernel's working directory, the page's own
    # directory as under Quarto.
    client = nbclient.NotebookClient(notebook, timeout=600, kernel_name="python3",
                                     resources={"metadata": {"path": str(page.parent)}})
    try:
        client.execute()
    except nbclient.exceptions.CellExecutionError as exc:
        pytest.fail(f"{name} raised in a cell:\n{exc}")


def test_every_page_named_in_the_navigation_exists():
    """A navbar entry pointing at a missing page renders as a dead link."""
    quarto = (ROOT / "_quarto.yml").read_text()
    page = r"\S+\.(?:qmd|py)"
    referenced = set(re.findall(rf"^\s*-\s+({page})\s*$", quarto, re.MULTILINE))
    referenced |= set(re.findall(rf"href:\s+({page})", quarto))
    # The project's `render:` list holds globs, which name no single page.
    referenced = {p for p in referenced if "*" not in p}
    missing = sorted(p for p in referenced if not (ROOT / p).exists())
    assert not missing, f"named in _quarto.yml but absent: {missing}"
