from __future__ import annotations

import base64
import contextlib
import csv
import difflib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from typing import Callable

from PIL import Image, ImageFilter, ImageOps

from . import caption as _caption, layout as _layout, ocr as _ocr, pipeline as _pipeline, shared as _shared, summary as _summary
from .caption import *
from .layout import *
from .ocr import *
from .pipeline import *
from .shared import *
from .summary import *

__all__ = [
    "analyze_image",
    "build_screen_description",
    "build_user_image_context",
    "get_local_image_capabilities",
    "has_caption_support",
    "has_local_image_support",
    "has_tesseract_ocr_support",
    "has_windows_ocr_support",
]

_SYNC_EXCLUDED = {
    "__all__",
    "_SYNC_EXCLUDED",
    "_SYNC_TARGETS",
    "_WRAPPER_ORIGINALS",
    "_WRAPPER_PASSTHROUGH",
    "_caption",
    "_layout",
    "_ocr",
    "_pipeline",
    "_shared",
    "_summary",
    "_sync_internal_globals",
}

_SYNC_TARGETS = (_caption, _layout, _ocr, _pipeline, _shared, _summary)


def _sync_internal_globals() -> None:
    excluded = set(_SYNC_EXCLUDED)
    for name, value in globals().items():
        if name in excluded or name.startswith("__"):
            continue
        synced_value = value
        if name in _WRAPPER_PASSTHROUGH and value is _WRAPPER_PASSTHROUGH[name]:
            synced_value = _WRAPPER_ORIGINALS[name]
        for target in _SYNC_TARGETS:
            if hasattr(target, name):
                setattr(target, name, synced_value)


def analyze_image(image_b64: str, *, debug_write: _DEBUG_WRITE | None = None) -> dict:
    _sync_internal_globals()
    return _pipeline.analyze_image(image_b64, debug_write=debug_write)


def build_user_image_context(
    images: list[str] | None,
    *,
    user_text: str = "",
    debug_write: _DEBUG_WRITE | None = None,
) -> str:
    _sync_internal_globals()
    return _pipeline.build_user_image_context(images, user_text=user_text, debug_write=debug_write)


def build_screen_description(image_b64: str, *, debug_write: _DEBUG_WRITE | None = None) -> str:
    _sync_internal_globals()
    return _pipeline.build_screen_description(image_b64, debug_write=debug_write)


def _run_local_ocr_with_backend(
    image,
    debug_write: _DEBUG_WRITE,
    *,
    include_layout: bool = False,
):
    _sync_internal_globals()
    return _ocr._run_local_ocr_with_backend(image, debug_write, include_layout=include_layout)


def _run_local_ocr(
    image,
    debug_write: _DEBUG_WRITE,
    *,
    include_layout: bool = False,
):
    _sync_internal_globals()
    return _ocr._run_local_ocr(image, debug_write, include_layout=include_layout)


def _run_windows_ocr(
    image,
    debug_write: _DEBUG_WRITE,
    *,
    include_layout: bool = False,
):
    _sync_internal_globals()
    return _ocr._run_windows_ocr(image, debug_write, include_layout=include_layout)


def _run_tesseract_ocr(
    image,
    debug_write: _DEBUG_WRITE,
    *,
    include_layout: bool = False,
):
    _sync_internal_globals()
    return _ocr._run_tesseract_ocr(image, debug_write, include_layout=include_layout)


def _should_retry_ocr(image, text: str) -> bool:
    _sync_internal_globals()
    return _ocr._should_retry_ocr(image, text)


def _build_ocr_retry_image(image):
    _sync_internal_globals()
    return _ocr._build_ocr_retry_image(image)


def _caption_image(image, debug_write: _DEBUG_WRITE, *, prompt: str = "") -> str:
    _sync_internal_globals()
    return _caption._caption_image(image, debug_write, prompt=prompt)


def _load_caption_backend(debug_write: _DEBUG_WRITE, *, blocking: bool = False):
    _sync_internal_globals()
    return _caption._load_caption_backend(debug_write, blocking=blocking)


def _ensure_caption_backend_loading(debug_write: _DEBUG_WRITE) -> None:
    _sync_internal_globals()
    return _caption._ensure_caption_backend_loading(debug_write)


def _get_caption_backend_state() -> dict[str, bool]:
    _sync_internal_globals()
    return _caption._get_caption_backend_state()


def _run_caption_analysis(image, ocr_text: str, debug_write: _DEBUG_WRITE) -> dict:
    _sync_internal_globals()
    return _caption._run_caption_analysis(image, ocr_text, debug_write)


def _normalize_ocr_lines(
    image,
    lines,
    *,
    source_size: tuple[int, int] | None = None,
):
    _sync_internal_globals()
    return _layout._normalize_ocr_lines(image, lines, source_size=source_size)


def _merge_ocr_lines(primary_lines, secondary_lines):
    _sync_internal_globals()
    return _layout._merge_ocr_lines(primary_lines, secondary_lines)


def _extract_chart_text_structure(image, ocr_lines):
    _sync_internal_globals()
    return _layout._extract_chart_text_structure(image, ocr_lines)


def _analyze_structured_layout(image, ocr_lines, ocr_text: str) -> dict:
    _sync_internal_globals()
    return _layout._analyze_structured_layout(image, ocr_lines, ocr_text)


def _analyze_chart_visual_pattern(image) -> dict:
    _sync_internal_globals()
    return _layout._analyze_chart_visual_pattern(image)


def _detect_visual_scene(image, *, caption: str, ocr_text: str, chart_visual: dict | None = None) -> str:
    _sync_internal_globals()
    return _layout._detect_visual_scene(image, caption=caption, ocr_text=ocr_text, chart_visual=chart_visual)


def _repair_short_ui_text(text: str, *, allow_url: bool = False) -> str:
    _sync_internal_globals()
    return _layout._repair_short_ui_text(text, allow_url=allow_url)


def _repair_url_like_text(text: str) -> str:
    _sync_internal_globals()
    return _layout._repair_url_like_text(text)


def _build_clean_visible_text(*, scene: str, layout: dict, chart_text: dict, ocr_text: str) -> str:
    _sync_internal_globals()
    return _layout._build_clean_visible_text(scene=scene, layout=layout, chart_text=chart_text, ocr_text=ocr_text)


def _should_attempt_caption(image, ocr_text: str) -> bool:
    _sync_internal_globals()
    return _summary._should_attempt_caption(image, ocr_text)


def _choose_caption_prompt(image, ocr_text: str) -> str:
    _sync_internal_globals()
    return _summary._choose_caption_prompt(image, ocr_text)


def _build_visual_summary(
    *,
    image,
    caption: str,
    ocr_text: str,
    scene: str | None = None,
    chart_visual: dict | None = None,
    chart_text: dict | None = None,
    layout: dict | None = None,
) -> str:
    _sync_internal_globals()
    return _summary._build_visual_summary(
        image=image,
        caption=caption,
        ocr_text=ocr_text,
        scene=scene,
        chart_visual=chart_visual,
        chart_text=chart_text,
        layout=layout,
    )


def get_local_image_capabilities() -> dict[str, bool]:
    _sync_internal_globals()
    return _ocr.get_local_image_capabilities()


def has_local_image_support() -> bool:
    _sync_internal_globals()
    return _ocr.has_local_image_support()


def has_windows_ocr_support() -> bool:
    _sync_internal_globals()
    return _ocr.has_windows_ocr_support()


def has_tesseract_ocr_support() -> bool:
    _sync_internal_globals()
    return _ocr.has_tesseract_ocr_support()


_WRAPPER_PASSTHROUGH = {
    "analyze_image": analyze_image,
    "build_screen_description": build_screen_description,
    "build_user_image_context": build_user_image_context,
    "get_local_image_capabilities": get_local_image_capabilities,
    "has_local_image_support": has_local_image_support,
    "has_tesseract_ocr_support": has_tesseract_ocr_support,
    "has_windows_ocr_support": has_windows_ocr_support,
    "_analyze_chart_visual_pattern": _analyze_chart_visual_pattern,
    "_analyze_structured_layout": _analyze_structured_layout,
    "_build_clean_visible_text": _build_clean_visible_text,
    "_build_ocr_retry_image": _build_ocr_retry_image,
    "_build_visual_summary": _build_visual_summary,
    "_caption_image": _caption_image,
    "_choose_caption_prompt": _choose_caption_prompt,
    "_detect_visual_scene": _detect_visual_scene,
    "_ensure_caption_backend_loading": _ensure_caption_backend_loading,
    "_extract_chart_text_structure": _extract_chart_text_structure,
    "_get_caption_backend_state": _get_caption_backend_state,
    "_load_caption_backend": _load_caption_backend,
    "_merge_ocr_lines": _merge_ocr_lines,
    "_normalize_ocr_lines": _normalize_ocr_lines,
    "_repair_short_ui_text": _repair_short_ui_text,
    "_repair_url_like_text": _repair_url_like_text,
    "_run_caption_analysis": _run_caption_analysis,
    "_run_local_ocr": _run_local_ocr,
    "_run_local_ocr_with_backend": _run_local_ocr_with_backend,
    "_run_tesseract_ocr": _run_tesseract_ocr,
    "_run_windows_ocr": _run_windows_ocr,
    "_should_attempt_caption": _should_attempt_caption,
    "_should_retry_ocr": _should_retry_ocr,
}

_WRAPPER_ORIGINALS = {
    "analyze_image": _pipeline.analyze_image,
    "build_screen_description": _pipeline.build_screen_description,
    "build_user_image_context": _pipeline.build_user_image_context,
    "get_local_image_capabilities": _ocr.get_local_image_capabilities,
    "has_local_image_support": _ocr.has_local_image_support,
    "has_tesseract_ocr_support": _ocr.has_tesseract_ocr_support,
    "has_windows_ocr_support": _ocr.has_windows_ocr_support,
    "_analyze_chart_visual_pattern": _layout._analyze_chart_visual_pattern,
    "_analyze_structured_layout": _layout._analyze_structured_layout,
    "_build_clean_visible_text": _layout._build_clean_visible_text,
    "_build_ocr_retry_image": _ocr._build_ocr_retry_image,
    "_build_visual_summary": _summary._build_visual_summary,
    "_caption_image": _caption._caption_image,
    "_choose_caption_prompt": _summary._choose_caption_prompt,
    "_detect_visual_scene": _layout._detect_visual_scene,
    "_ensure_caption_backend_loading": _caption._ensure_caption_backend_loading,
    "_extract_chart_text_structure": _layout._extract_chart_text_structure,
    "_get_caption_backend_state": _caption._get_caption_backend_state,
    "_load_caption_backend": _caption._load_caption_backend,
    "_merge_ocr_lines": _layout._merge_ocr_lines,
    "_normalize_ocr_lines": _layout._normalize_ocr_lines,
    "_repair_short_ui_text": _layout._repair_short_ui_text,
    "_repair_url_like_text": _layout._repair_url_like_text,
    "_run_caption_analysis": _caption._run_caption_analysis,
    "_run_local_ocr": _ocr._run_local_ocr,
    "_run_local_ocr_with_backend": _ocr._run_local_ocr_with_backend,
    "_run_tesseract_ocr": _ocr._run_tesseract_ocr,
    "_run_windows_ocr": _ocr._run_windows_ocr,
    "_should_attempt_caption": _summary._should_attempt_caption,
    "_should_retry_ocr": _ocr._should_retry_ocr,
}
