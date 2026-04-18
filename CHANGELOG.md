# Changelog

## Unreleased

- split the project into internal `ocr`, `layout`, `summary`, `caption`, and `pipeline` modules behind a top-level compatibility facade
- introduced `vision_local_core/shared.py` for shared runtime state, keyword tables, and cross-module text helpers so the active modules rely less on `_legacy.py`
- moved the high-level analysis pipeline and prompt-context builders into `vision_local_core/pipeline.py`
- moved layout parsing, scene detection, and chart heuristics into `vision_local_core/layout.py`
- moved summary generation and caption-decision helpers into `vision_local_core/summary.py`
- moved the caption runtime implementation and caption analysis flow into `vision_local_core/caption.py`
- moved the OCR runtime implementation and OCR analysis flow into `vision_local_core/ocr.py` while keeping the top-level patching surface stable for tests
- exposed the OCR backend used in `analyze_image()` results and analysis timing debug output
- added a GitHub Actions OCR integration job that installs Tesseract and runs real fallback-path tests
- added an OCR backend abstraction with Tesseract CLI fallback for non-Windows environments
- exposed Tesseract OCR capability probes and configuration knobs for backend selection, language choice, and page segmentation mode
- added golden regression coverage for mixed-language browser and chat screens, low-resolution chat layouts, and downward-trend chart cases
- broadened browser, chat, chart, and settings keyword handling for mixed Chinese-English screenshots
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
