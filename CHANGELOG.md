# Changelog

## Unreleased

- stabilized the `analyze_image()` failure payload so callers always receive the same top-level fields
- refactored the analysis pipeline helpers to keep OCR and caption flow easier to maintain
- added README visual assets and a more polished GitHub landing layout
- improved the GitHub-facing README structure with badges, clearer install paths, and a stronger project overview
- added issue templates for bug reports and feature requests

## 0.1.0 - 2026-04-18

Initial standalone extraction.

- added `vision_local.py` as the primary local image-analysis module
- kept direct entry points for analysis, screen description, and prompt-context generation
- included copied regression tests for OCR cleanup, structured layout inference, and chart heuristics
- added standalone packaging metadata with `pyproject.toml`
- added MIT license and repository-ready README
- renamed standalone configuration to `VISION_LOCAL_CONTEXT_*`
- added explicit capability probes for Windows OCR and optional caption support
