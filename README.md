# Vision Local Context

[![CI](https://img.shields.io/github/actions/workflow/status/agent-wrangler/vision-local-context/ci.yml?branch=main&label=CI)](https://github.com/agent-wrangler/vision-local-context/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/agent-wrangler/vision-local-context)](https://github.com/agent-wrangler/vision-local-context/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)](https://github.com/agent-wrangler/vision-local-context)

Windows-first local screenshot understanding for OCR, UI layout extraction, and LLM-ready prompt context.

`vision-local-context` sits in the practical middle layer between raw screenshots and a chat model. It is built for workflows where you do not want to make a multimodal API call for every image turn, but you still want useful local understanding of browser pages, chat windows, settings screens, dashboards, and documents.

## Why This Exists

- turn screenshots into structured prompt context instead of raw OCR dumps
- recover useful UI labels from noisy OCR output
- detect common screen types such as browser, chat, chart, settings, and document
- extract browser-style and chat-style layout hints for downstream tools or agents
- keep the integration surface small: one module, a few entry points, and plain Python dictionaries

## Core Entry Points

- `analyze_image(image_b64)` returns structured analysis for one image
- `build_user_image_context(images, user_text="")` builds a prompt-ready context block for one or more images
- `build_screen_description(image_b64)` returns a compact one-line description

The module also exposes capability probes:

- `has_windows_ocr_support()`
- `has_caption_support()`
- `get_local_image_capabilities()`
- `has_local_image_support()`

## Typical Output

`analyze_image(image_b64)` returns a dictionary shaped roughly like this:

```python
{
    "ok": True,
    "scene": "browser",
    "summary": "This appears to be a browser or website page.",
    "visible_text": "https://example.com/login Example Login Email Password Sign in",
    "layout": {
        "kind": "browser",
        "address_bar": "https://example.com/login",
        "page_title": "Example Login",
        "field_labels": ["Email", "Password", "Sign in"],
    },
}
```

Typical fields include:

- scene type such as `browser`, `chat`, `chart`, `settings`, or `document`
- cleaned visible text
- layout details like address bar, page title, sidebar labels, and input hint
- chart details such as title, axis labels, and trend hints
- a plain-language summary that can be injected into a prompt

## Install

Clone the repository and install from source:

```bash
git clone https://github.com/agent-wrangler/vision-local-context.git
cd vision-local-context
pip install .
```

For optional caption-model support:

```bash
pip install ".[caption]"
```

For tests:

```bash
pip install ".[test]"
python -m pytest -q
```

For local development and release checks:

```bash
pip install -e ".[dev]"
python -m pytest
python -m build
```

## Quick Start

```python
import base64
from pathlib import Path

from vision_local import analyze_image, build_user_image_context

image_b64 = base64.b64encode(Path("example.png").read_bytes()).decode("ascii")

analysis = analyze_image(image_b64)
print(analysis["scene"])
print(analysis["summary"])

context = build_user_image_context(
    [image_b64],
    user_text="What is on this screen?",
)
print(context)
```

## Platform Notes

- OCR currently uses Windows OCR through PowerShell and Windows Runtime APIs.
- Optional caption generation uses BLIP through `transformers` and `torch`.
- The package can still be imported and tested on non-Windows platforms.
- Full OCR-driven analysis is Windows-focused today.

## Configuration

Optional environment variables:

- `VISION_LOCAL_CONTEXT_OCR_TIMEOUT_SECONDS`
- `VISION_LOCAL_CONTEXT_CAPTION_BLOCKING`
- `VISION_LOCAL_CONTEXT_CAPTION_ALLOW_DOWNLOAD`
- `VISION_LOCAL_CONTEXT_CAPTION_MODEL`

## Project Status

This is an early standalone extraction of a local vision-context layer. The current scope is intentionally narrow: keep the core analysis module clean, testable, and easy to embed into another agent or chat runtime.

Near-term priorities:

- continue hardening OCR cleanup and layout extraction heuristics
- expand screenshot coverage with more regression fixtures
- keep the public API small and stable

## Repository Layout

- `vision_local.py`: main module
- `tests/test_vision_local_context.py`: regression tests for OCR cleanup, layout inference, and chart heuristics
- `.github/workflows/ci.yml`: test and packaging checks for pushes and pull requests
- `CHANGELOG.md`: release notes

## Development

1. Create a virtual environment with Python 3.10 or newer.
2. Install editable dependencies with `pip install -e ".[dev]"`.
3. Run `python -m pytest` before opening a pull request.
4. Run `python -m build` before tagging a release.

GitHub Actions runs the test suite on Windows and Linux and performs a packaging check on every push and pull request.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for local setup, pull-request expectations, and release checks.

## License

MIT
