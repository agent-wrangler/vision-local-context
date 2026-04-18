# Contributing

Thanks for helping improve `vision-local-context`.

## Local Setup

1. Use Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install the project in editable mode:

```bash
pip install -e .[dev]
```

## Before Opening a Pull Request

Run the local checks:

```bash
python -m pytest
python -m build
```

## Scope and Style

- Keep the standalone module easy to embed in other projects.
- Prefer small, focused changes with regression tests when behavior changes.
- Preserve the Windows OCR capability checks so the module degrades cleanly on other platforms.
- Update `README.md` and `CHANGELOG.md` when public behavior or install steps change.

## Pull Requests

- Explain the user-visible change and why it matters.
- Call out any platform-specific behavior, especially around Windows OCR or optional caption support.
- Include tests for bug fixes or new heuristics when practical.
