from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from typing import Callable

from PIL import Image

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
_STOP_LABEL_WORDS = {
    "page",
    "screen",
    "window",
    "user",
    "showing",
    "computer",
    "desktop",
    "interface",
    "button",
    "mode",
    "panel",
    "menu",
}
_SETTINGS_KEYWORDS = {
    "setting",
    "settings",
    "\u8bbe\u7f6e",
    "network",
    "\u7f51\u7edc",
    "bluetooth",
    "\u84dd\u7259",
    "display",
    "\u663e\u793a",
    "battery",
    "\u7535\u6c60",
    "privacy",
    "\u9690\u79c1",
    "wifi",
    "wi-fi",
    "brightness",
    "\u4eae\u5ea6",
    "power mode",
    "volume",
    "\u97f3\u91cf",
    "audio",
    "\u58f0\u97f3",
}
_CHART_KEYWORDS = {
    "chart",
    "\u56fe\u8868",
    "graph",
    "dashboard",
    "\u4eea\u8868\u76d8",
    "trend",
    "\u8d8b\u52bf",
    "sales",
    "revenue",
    "\u8425\u6536",
    "\u6536\u5165",
    "growth",
    "legend",
    "axis",
    "metric",
}
_BROWSER_KEYWORDS = {
    "browser",
    "\u6d4f\u89c8\u5668",
    "chrome",
    "edge",
    "search",
    "\u641c\u7d22",
    "tab",
    "login",
    "\u767b\u5f55",
    "\u767b\u5165",
    "sign in",
    "website",
    "http",
    "www",
    "address bar",
    "\u5730\u5740\u680f",
}
_CHAT_KEYWORDS = {
    "chat",
    "\u804a\u5929",
    "message",
    "messages",
    "\u6d88\u606f",
    "reply",
    "\u56de\u590d",
    "assistant",
    "conversation",
    "\u5bf9\u8bdd",
    "send",
    "\u53d1\u9001",
    "typing",
    "\u8f93\u5165",
    "wechat",
    "slack",
    "discord",
    "telegram",
}
_BROWSER_FIELD_TOKENS = (
    "email",
    "password",
    "username",
    "search",
    "login",
    "sign in",
    "submit",
    "\u90ae\u7bb1",
    "\u90ae\u4ef6",
    "\u5bc6\u7801",
    "\u7528\u6237\u540d",
    "\u641c\u7d22",
    "\u767b\u5f55",
    "\u767b\u5165",
    "\u63d0\u4ea4",
)
_BROWSER_TITLE_EXCLUDE_TOKENS = (
    "email",
    "password",
    "username",
    "search",
    "sign in",
    "submit",
    "\u90ae\u7bb1",
    "\u90ae\u4ef6",
    "\u5bc6\u7801",
    "\u7528\u6237\u540d",
    "\u641c\u7d22",
    "\u63d0\u4ea4",
)
_CHAT_INPUT_TOKENS = (
    "send",
    "type",
    "message",
    "reply",
    "input",
    "write",
    "chat",
    "\u53d1\u9001",
    "\u8f93\u5165",
    "\u6d88\u606f",
    "\u56de\u590d",
)
_COMMON_UI_CANONICALS = (
    "Example",
    "Login",
    "Example Login",
    "Log in",
    "Sign in",
    "Sign up",
    "Email",
    "Password",
    "Username",
    "Search",
    "Settings",
    "General",
    "Design",
    "Send",
    "Reply",
    "Message",
    "Messages",
    "Type a message",
    "Dashboard",
    "Chat",
)


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
