# Vision Local Context

Local screenshot and image-understanding helpers for turning uploaded images into structured, LLM-ready context.

This project focuses on the practical middle layer between raw screenshots and a chat model.

It is built for cases where you do not want to rely on a native multimodal API call for every image turn, but you still want useful local understanding of screenshots, browser UIs, chat windows, settings pages, and dashboards.

## Features

The current module provides:

- OCR with a Windows-native local backend
- OCR cleanup and short-label repair
- screenshot scene detection
- structured layout extraction for browser and chat UIs
- chart heuristics for line and bar dashboards
- text summaries that can be injected into an LLM prompt
- prompt-ready context assembly for one or more uploaded images

## Capability Probes

The module exposes a few simple capability checks:

- `has_windows_ocr_support()` for the full OCR and layout-extraction path
- `has_caption_support()` for optional BLIP caption support
- `get_local_image_capabilities()` for a combined view
- `has_local_image_support()` as a conservative shortcut for the full OCR-driven path

## What It Returns

The core entry points are:

- `analyze_image(image_b64)` for structured analysis
- `build_user_image_context(images, user_text="")` for a ready-to-inject context block
- `build_screen_description(image_b64)` for a compact one-line screen summary

Typical output includes:

- scene type such as `browser`, `chat`, `chart`, `settings`, or `document`
- cleaned visible text
- layout details like address bar, page title, sidebar labels, and input hint
- chart details such as title, axis labels, and trend hints
- a plain-language summary for downstream prompting

## Example Analysis

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

## Install

```bash
pip install -e .
```

With optional caption-model support:

```bash
pip install -e .[caption]
```

For tests:

```bash
pip install -e .[test]
pytest -q
```

For local development and release checks:

```bash
pip install -e .[dev]
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

context = build_user_image_context([image_b64], user_text="What is on this screen?")
print(context)
```

## Platform Notes

- OCR currently uses Windows OCR through PowerShell and Windows Runtime APIs.
- Optional caption generation uses BLIP through `transformers` and `torch`.
- The module still works without caption support, but visual summaries may rely more heavily on OCR and layout inference.
- Full OCR analysis is Windows-focused today, but the package can still be imported and tested on non-Windows platforms.

## Configuration

Optional environment variables:

- `VISION_LOCAL_CONTEXT_OCR_TIMEOUT_SECONDS`
- `VISION_LOCAL_CONTEXT_CAPTION_BLOCKING`
- `VISION_LOCAL_CONTEXT_CAPTION_ALLOW_DOWNLOAD`
- `VISION_LOCAL_CONTEXT_CAPTION_MODEL`

## Status

This is an early extracted standalone version of the local vision-context layer. The current scope is intentionally narrow: keep the core analysis module clean, testable, and easy to embed into another agent or chat runtime.

## Repository Layout

- `vision_local.py`: main module
- `tests/test_vision_local_context.py`: copied regression tests for the standalone module
- `CHANGELOG.md`: release notes for standalone extraction milestones

## Development

1. Create a virtual environment with Python 3.10+.
2. Install editable dependencies with `pip install -e .[dev]`.
3. Run `python -m pytest` before opening a pull request.
4. Run `python -m build` before tagging a release.

GitHub Actions runs the test suite on Windows and Linux and performs a packaging check on every push and pull request.

## License

MIT
