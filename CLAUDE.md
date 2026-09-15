See [AGENTS.md](AGENTS.md). The working rules for this repository live in that
one file, so the two cannot drift apart.

## Rendering the site locally

Quarto runs the page code with the first `python3` it finds, and on phoenix
that Python has no `yaml` module. Set `QUARTO_PYTHON` to the project Python
before a render:

```bash
export QUARTO_PYTHON=~/Workplace/python3.12/bin/python3.12
quarto render
```

If `QUARTO_PYTHON` is not set, the render stops at once with
`ModuleNotFoundError: No module named 'yaml'`. A render writes `_site/`,
`.quarto/` and `_freeze/`, and all three are gitignored.
