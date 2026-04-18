from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from typing import Callable

from PIL import Image

from ._legacy import (
    _BROWSER_FIELD_TOKENS,
    _BROWSER_KEYWORDS,
    _BROWSER_TITLE_EXCLUDE_TOKENS,
    _CHART_KEYWORDS,
    _CHAT_INPUT_TOKENS,
    _CHAT_KEYWORDS,
    _COMMON_UI_CANONICALS,
    _SETTINGS_KEYWORDS,
    _STOP_LABEL_WORDS,
)

_DEBUG_WRITE = Callable[[str, dict], None]
_CACHE_LIMIT = 128
_ANALYSIS_CACHE: OrderedDict[str, dict] = OrderedDict()
_CACHE_LOCK = threading.Lock()

_CAPTION_LOCK = threading.Lock()
_CAPTION_BACKEND: tuple[object, object] | None = None
_CAPTION_LOAD_ATTEMPTED = False
_CAPTION_LOAD_ERROR = ""
_CAPTION_LOADING = False

_RESAMPLING = getattr(Image, "Resampling", Image)


def _normalize_text(text: str, *, limit: int = 900) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return ""
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 3, 0)].rstrip() + "..."


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    raw = str(os.environ.get(name, "") or "").strip()
    try:
        value = int(raw)
    except Exception:
        value = default
    if value < minimum:
        value = minimum
    if maximum is not None and value > maximum:
        value = maximum
    return value


def _extract_readable_labels(text: str, *, limit: int = 6) -> list[str]:
    normalized = _normalize_text(text, limit=1200)
    if not normalized:
        return []
    labels: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", normalized):
        lowered = raw.lower()
        if lowered in _STOP_LABEL_WORDS:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        labels.append(raw)
        if len(labels) >= limit:
            break
    return labels


def _extract_numeric_markers(text: str, *, limit: int = 8) -> list[str]:
    normalized = _normalize_text(text, limit=1200)
    if not normalized:
        return []
    markers: list[str] = []
    seen: set[str] = set()
    patterns = (
        r"\bQ[1-4]\b",
        r"[+-]?\d+(?:\.\d+)?\s*%",
        r"\d+(?:\.\d+)?\s*[kKmMgG]\b",
        r"\b\d{2,4}\b",
    )
    for pattern in patterns:
        for raw in re.findall(pattern, normalized, flags=re.IGNORECASE):
            marker = str(raw).strip()
            key = marker.lower()
            if not marker or key in seen:
                continue
            seen.add(key)
            markers.append(marker)
            if len(markers) >= limit:
                return markers
    return markers


def _looks_like_url_or_query(text: str) -> bool:
    lowered = _normalize_text(text, limit=240).lower()
    if not lowered:
        return False
    if any(token in lowered for token in ("http", "www.", ".com", ".cn", ".net", ".io", ".ai", "://")):
        return True
    if "/" in lowered and "." in lowered:
        return True
    if "\\" in lowered and "." in lowered:
        return True
    return lowered.count(".") >= 2 and " " not in lowered


def _ocr_signal_score(text: str) -> int:
    normalized = _normalize_text(text, limit=1600)
    if not normalized:
        return 0
    latin_words = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", normalized)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", normalized)
    digits = re.findall(r"\d", normalized)
    return len(normalized) + len(latin_words) * 9 + len(cjk_chars) * 4 + len(digits) * 2


def _is_low_signal_ocr(text: str) -> bool:
    normalized = _normalize_text(text, limit=1600)
    if not normalized:
        return True
    if len(re.findall(r"[\u4e00-\u9fff]", normalized)) >= 6:
        return False
    latin_words = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", normalized)
    if len(latin_words) >= 5 and len(normalized) >= 32:
        return False
    return _ocr_signal_score(normalized) < 72


def _ocr_line_text_quality(text: str) -> int:
    normalized = _normalize_text(text, limit=240)
    if not normalized:
        return -1000
    latin = len(re.findall(r"[A-Za-z]", normalized))
    cjk = len(re.findall(r"[\u4e00-\u9fff]", normalized))
    digits = len(re.findall(r"\d", normalized))
    weird = len(re.findall(r"[^A-Za-z0-9\s:/._%+\-()\u4e00-\u9fff]", normalized))
    score = len(normalized) + latin * 4 + cjk * 4 + digits * 2 - weird * 6
    if _looks_like_url_or_query(normalized):
        score += 18
    if latin + cjk <= 1 and digits <= 1:
        score -= 20
    return score


__all__ = [
    "_ANALYSIS_CACHE",
    "_BROWSER_FIELD_TOKENS",
    "_BROWSER_KEYWORDS",
    "_BROWSER_TITLE_EXCLUDE_TOKENS",
    "_CACHE_LIMIT",
    "_CACHE_LOCK",
    "_CAPTION_BACKEND",
    "_CAPTION_LOAD_ATTEMPTED",
    "_CAPTION_LOAD_ERROR",
    "_CAPTION_LOADING",
    "_CAPTION_LOCK",
    "_CHART_KEYWORDS",
    "_CHAT_INPUT_TOKENS",
    "_CHAT_KEYWORDS",
    "_COMMON_UI_CANONICALS",
    "_DEBUG_WRITE",
    "_RESAMPLING",
    "_SETTINGS_KEYWORDS",
    "_STOP_LABEL_WORDS",
    "_env_flag",
    "_env_int",
    "_extract_numeric_markers",
    "_extract_readable_labels",
    "_is_low_signal_ocr",
    "_looks_like_url_or_query",
    "_normalize_text",
    "_ocr_line_text_quality",
    "_ocr_signal_score",
]
