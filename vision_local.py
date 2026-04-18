from __future__ import annotations

import base64
import contextlib
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
    "network",
    "bluetooth",
    "display",
    "battery",
    "privacy",
    "wifi",
    "wi-fi",
    "brightness",
    "power mode",
    "volume",
    "audio",
}
_CHART_KEYWORDS = {
    "chart",
    "graph",
    "dashboard",
    "trend",
    "sales",
    "revenue",
    "growth",
    "legend",
    "axis",
    "metric",
}
_BROWSER_KEYWORDS = {
    "browser",
    "chrome",
    "edge",
    "search",
    "tab",
    "login",
    "sign in",
    "website",
    "http",
    "www",
    "address bar",
}
_CHAT_KEYWORDS = {
    "chat",
    "message",
    "messages",
    "reply",
    "assistant",
    "conversation",
    "send",
    "typing",
    "wechat",
    "slack",
    "discord",
    "telegram",
}
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


def _noop_debug(_stage: str, _data: dict) -> None:
    return None


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


def _ocr_timeout_seconds() -> int:
    return _env_int("AARONCORE_LOCAL_OCR_TIMEOUT_SECONDS", 8, minimum=3, maximum=20)


def _caption_blocking_enabled() -> bool:
    return _env_flag("AARONCORE_LOCAL_CAPTION_BLOCKING", default=False)


def _caption_allow_download() -> bool:
    return _env_flag("AARONCORE_LOCAL_CAPTION_ALLOW_DOWNLOAD", default=False)


def _decode_image(image_b64: str) -> tuple[Image.Image | None, bytes]:
    text = str(image_b64 or "").strip()
    if not text:
        return None, b""
    if text.startswith("data:"):
        _, _, text = text.partition(",")
    try:
        raw = base64.b64decode(text, validate=False)
    except Exception:
        return None, b""
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None, b""
    return image, raw


def _cache_get(digest: str) -> dict | None:
    if not digest:
        return None
    with _CACHE_LOCK:
        cached = _ANALYSIS_CACHE.get(digest)
        if cached is None:
            return None
        _ANALYSIS_CACHE.move_to_end(digest)
        return dict(cached)


def _cache_put(digest: str, analysis: dict) -> None:
    if not digest:
        return
    with _CACHE_LOCK:
        _ANALYSIS_CACHE[digest] = dict(analysis)
        _ANALYSIS_CACHE.move_to_end(digest)
        while len(_ANALYSIS_CACHE) > _CACHE_LIMIT:
            _ANALYSIS_CACHE.popitem(last=False)


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


def _build_ocr_retry_image(image: Image.Image) -> Image.Image:
    max_width = 2560
    max_height = 1920
    scale = min(2.0, max_width / max(image.width, 1), max_height / max(image.height, 1))
    processed = ImageOps.autocontrast(ImageOps.grayscale(image))
    if scale > 1.05:
        new_size = (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        )
        processed = processed.resize(new_size, _RESAMPLING.LANCZOS)
    processed = processed.filter(ImageFilter.SHARPEN)
    return processed.convert("RGB")


def _should_retry_ocr(image: Image.Image, text: str) -> bool:
    normalized = _normalize_text(text, limit=1200)
    if _is_low_signal_ocr(normalized):
        return True
    if image.width < 900 or image.height < 500:
        return False
    if len(normalized) >= 96:
        return False
    return len(_extract_readable_labels(normalized, limit=8)) < 6


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


def _count_keyword_hits(text: str, keywords: set[str]) -> tuple[int, set[str]]:
    lowered = _normalize_text(text, limit=1600).lower()
    hits = {keyword for keyword in keywords if keyword in lowered}
    return len(hits), hits


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


def _stabilize_layout_text(text: str) -> str:
    normalized = _normalize_text(text, limit=240)
    if not normalized:
        return ""
    normalized = normalized.replace("\\", "/")
    normalized = re.sub(r"^(https?)(//)", r"\1://", normalized, flags=re.IGNORECASE)
    return normalized


def _ambiguous_signature(text: str) -> str:
    normalized = _normalize_text(text, limit=240).lower()
    if not normalized:
        return ""
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", normalized)
    return normalized.translate(
        str.maketrans(
            {
                "i": "1",
                "l": "1",
                "1": "1",
                "o": "0",
                "0": "0",
                "s": "5",
                "5": "5",
                "b": "8",
                "8": "8",
            }
        )
    )


def _match_common_ui_text(text: str) -> str:
    signature = _ambiguous_signature(text)
    if not signature:
        return ""
    best_match = ""
    best_ratio = 0.0
    for canonical in _COMMON_UI_CANONICALS:
        canonical_signature = _ambiguous_signature(canonical)
        if not canonical_signature:
            continue
        if signature == canonical_signature:
            return canonical
        if abs(len(signature) - len(canonical_signature)) > max(2, len(canonical_signature) // 3):
            continue
        ratio = difflib.SequenceMatcher(None, signature, canonical_signature).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = canonical
    threshold = 0.96
    if len(signature) >= 10:
        threshold = 0.84
    elif len(signature) >= 7:
        threshold = 0.88
    elif len(signature) >= 5:
        threshold = 0.9
    if best_ratio >= threshold:
        return best_match
    return ""


def _repair_url_like_text(text: str) -> str:
    normalized = _stabilize_layout_text(text)
    if not normalized:
        return ""
    lowered = normalized.lower()
    if lowered.startswith("https//"):
        normalized = "https://" + normalized[7:]
    elif lowered.startswith("http//"):
        normalized = "http://" + normalized[6:]
    elif lowered.startswith("https:/") and not lowered.startswith("https://"):
        normalized = "https://" + normalized[7:]
    elif lowered.startswith("http:/") and not lowered.startswith("http://"):
        normalized = "http://" + normalized[6:]
    match = re.match(r"^(https?://)?([^/\s]+)(/[^?\s#]*)?(.*)$", normalized, flags=re.IGNORECASE)
    if not match:
        return normalized
    prefix = match.group(1) or ""
    domain = match.group(2) or ""
    path = match.group(3) or ""
    suffix = match.group(4) or ""
    if path:
        repaired_segments: list[str] = []
        for raw_segment in path.split("/"):
            segment = str(raw_segment or "").strip()
            if not segment:
                repaired_segments.append("")
                continue
            canonical = _match_common_ui_text(segment.replace("-", " ").replace("_", " "))
            if canonical:
                segment = canonical.lower().replace(" ", "")
            repaired_segments.append(segment)
        path = "/".join(repaired_segments)
    return f"{prefix}{domain}{path}{suffix}"


def _repair_short_ui_text(text: str, *, allow_url: bool = False) -> str:
    normalized = _stabilize_layout_text(text)
    if not normalized:
        return ""
    if allow_url or _looks_like_url_or_query(normalized):
        return _repair_url_like_text(normalized)
    if len(normalized) > 80:
        return normalized
    canonical = _match_common_ui_text(normalized)
    if canonical:
        return canonical
    return normalized


def _dedupe_texts(values: list[str] | tuple[str, ...], *, limit: int | None = None) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for raw in values or []:
        text = _normalize_text(raw, limit=240)
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        deduped.append(text)
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


def _normalize_short_text_list(values: list[str] | tuple[str, ...], *, limit: int = 6) -> list[str]:
    normalized = [_repair_short_ui_text(raw) for raw in values or []]
    return _dedupe_texts(normalized, limit=limit)


def _normalize_ocr_lines(
    image: Image.Image,
    lines: list[dict] | None,
    *,
    source_size: tuple[int, int] | None = None,
) -> list[dict]:
    normalized_lines: list[dict] = []
    width = max(int(image.width), 1)
    height = max(int(image.height), 1)
    source_width = max(int((source_size or (width, height))[0]), 1)
    source_height = max(int((source_size or (width, height))[1]), 1)
    scale_x = width / float(source_width)
    scale_y = height / float(source_height)
    for raw in lines or []:
        text = _normalize_text(raw.get("text", ""), limit=220)
        if not text:
            continue
        try:
            x = int(round(float(raw.get("x", 0) or 0) * scale_x))
            y = int(round(float(raw.get("y", 0) or 0) * scale_y))
            line_width = int(round(float(raw.get("width", 0) or 0) * scale_x))
            line_height = int(round(float(raw.get("height", 0) or 0) * scale_y))
        except Exception:
            continue
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        if line_width <= 0:
            line_width = min(width - x, max(12, len(text) * 8))
        if line_height <= 0:
            line_height = min(height - y, 12)
        line_width = max(1, min(width - x, line_width))
        line_height = max(1, min(height - y, line_height))
        normalized_lines.append(
            {
                "text": text,
                "x": x,
                "y": y,
                "width": line_width,
                "height": line_height,
                "right": x + line_width,
                "bottom": y + line_height,
            }
        )
    normalized_lines.sort(key=lambda item: (item["y"], item["x"]))
    return normalized_lines


def _merge_ocr_lines(primary_lines: list[dict], secondary_lines: list[dict]) -> list[dict]:
    merged = [dict(item) for item in (primary_lines or [])]
    for candidate in secondary_lines or []:
        matched_index = -1
        best_score = -1.0
        for index, existing in enumerate(merged):
            x_overlap = max(
                0,
                min(existing["right"], candidate["right"]) - max(existing["x"], candidate["x"]),
            )
            y_overlap = max(
                0,
                min(existing["bottom"], candidate["bottom"]) - max(existing["y"], candidate["y"]),
            )
            overlap_area = x_overlap * y_overlap
            existing_area = max(existing["width"] * existing["height"], 1)
            candidate_area = max(candidate["width"] * candidate["height"], 1)
            overlap_ratio = overlap_area / float(max(existing_area, candidate_area))
            center_dx = abs((existing["x"] + existing["right"]) - (candidate["x"] + candidate["right"])) / 2.0
            center_dy = abs((existing["y"] + existing["bottom"]) - (candidate["y"] + candidate["bottom"])) / 2.0
            if overlap_ratio > 0.12:
                score = overlap_ratio
            elif center_dx <= max(24, max(existing["width"], candidate["width"]) * 0.75) and center_dy <= max(
                16, max(existing["height"], candidate["height"]) * 2.0
            ):
                score = 0.1 + (1.0 / (1.0 + center_dx + center_dy))
            else:
                continue
            if score > best_score:
                best_score = score
                matched_index = index
        if matched_index >= 0:
            current = dict(merged[matched_index])
            if _ocr_line_text_quality(candidate["text"]) > _ocr_line_text_quality(current["text"]):
                current["text"] = candidate["text"]
            current["x"] = min(current["x"], candidate["x"])
            current["y"] = min(current["y"], candidate["y"])
            current["right"] = max(current["right"], candidate["right"])
            current["bottom"] = max(current["bottom"], candidate["bottom"])
            current["width"] = current["right"] - current["x"]
            current["height"] = current["bottom"] - current["y"]
            merged[matched_index] = current
        else:
            merged.append(dict(candidate))
    merged.sort(key=lambda item: (item["y"], item["x"]))
    return merged


def _extract_chart_text_structure(image: Image.Image, ocr_lines: list[dict]) -> dict:
    if not ocr_lines:
        return {}
    width = max(int(image.width), 1)
    height = max(int(image.height), 1)

    def _line_texts(candidates: list[dict], *, limit: int = 6) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            text = _normalize_text(item.get("text", ""), limit=80)
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            values.append(text)
            if len(values) >= limit:
                break
        return values

    title_candidates = [
        line
        for line in ocr_lines
        if line["y"] <= int(height * 0.24)
        and line["x"] <= int(width * 0.72)
        and len(re.findall(r"[A-Za-z\u4e00-\u9fff]", line["text"])) >= 4
        and not _looks_like_url_or_query(line["text"])
    ]
    title = ""
    if title_candidates:
        title_line = max(
            title_candidates,
            key=lambda line: (min(line["width"], int(width * 0.65)), -line["y"], len(line["text"])),
        )
        title = title_line["text"]

    x_axis_candidates = [
        line
        for line in ocr_lines
        if line["y"] >= int(height * 0.68)
        and line["x"] >= int(width * 0.12)
        and len(line["text"]) <= 18
    ]
    x_axis_candidates.sort(key=lambda line: line["x"])

    y_axis_candidates = [
        line
        for line in ocr_lines
        if line["x"] <= int(width * 0.18)
        and int(height * 0.14) <= line["y"] <= int(height * 0.86)
        and re.search(r"\d", line["text"])
    ]
    y_axis_candidates.sort(key=lambda line: line["y"])

    legend_candidates = [
        line
        for line in ocr_lines
        if line["x"] >= int(width * 0.54)
        and line["y"] <= int(height * 0.34)
        and line["text"] != title
        and not re.fullmatch(r"[Qq]?\d+(?:\.\d+)?%?", line["text"])
    ]
    legend_candidates.sort(key=lambda line: (line["y"], line["x"]))

    structure = {
        "title": title,
        "x_axis_labels": _line_texts(x_axis_candidates, limit=6),
        "y_axis_labels": _line_texts(y_axis_candidates, limit=6),
        "legend_labels": _line_texts(legend_candidates, limit=4),
    }
    if any(structure.values()):
        return structure
    return {}


def _analyze_structured_layout(image: Image.Image, ocr_lines: list[dict], ocr_text: str) -> dict:
    if not ocr_lines:
        return {}
    width = max(int(image.width), 1)
    height = max(int(image.height), 1)
    lowered_text = _normalize_text(ocr_text, limit=1600).lower()
    top_lines = [line for line in ocr_lines if line["y"] <= int(height * 0.22)]
    body_lines = [
        line
        for line in ocr_lines
        if int(height * 0.10) <= line["y"] <= int(height * 0.84)
    ]
    bottom_lines = [line for line in ocr_lines if line["y"] >= int(height * 0.76)]

    address_candidates = [
        line
        for line in top_lines
        if _looks_like_url_or_query(line["text"]) or "search" in line["text"].lower()
    ]
    browser_title_candidates = [
        line
        for line in ocr_lines
        if line not in address_candidates
        and line["y"] <= int(height * 0.42)
        and int(width * 0.15) <= line["x"] <= int(width * 0.82)
        and len(re.findall(r"[A-Za-z\u4e00-\u9fff]", line["text"])) >= 3
        and not any(
            token in line["text"].lower()
            for token in ("email", "password", "username", "search", "sign in", "submit")
        )
    ]
    field_candidates = [
        line
        for line in body_lines
        if any(
            token in line["text"].lower()
            for token in ("email", "password", "username", "search", "login", "sign in", "submit")
        )
    ]
    if address_candidates or ("browser" in lowered_text and field_candidates):
        address_text = max(address_candidates, key=lambda line: line["width"])["text"] if address_candidates else ""
        page_title = (
            max(browser_title_candidates, key=lambda line: (len(line["text"]), line["width"]))["text"]
            if browser_title_candidates
            else ""
        )
        field_labels = _normalize_short_text_list([line["text"] for line in field_candidates[:4]], limit=4)
        return {
            "kind": "browser",
            "address_bar": _repair_short_ui_text(address_text, allow_url=True),
            "page_title": _repair_short_ui_text(page_title),
            "field_labels": field_labels,
        }

    left_lines = [
        line
        for line in body_lines
        if line["x"] <= int(width * 0.42)
        and line["right"] <= int(width * 0.62)
        and line["width"] <= int(width * 0.44)
    ]
    right_lines = [
        line
        for line in body_lines
        if line["x"] >= int(width * 0.48)
        and line["width"] <= int(width * 0.40)
    ]
    input_candidates = [
        line
        for line in bottom_lines
        if line["width"] >= max(40, int(width * 0.05))
        and any(
            token in line["text"].lower()
            for token in ("send", "type", "message", "reply", "input", "write", "chat")
        )
    ]
    sidebar_lines = [
        line
        for line in body_lines
        if line["x"] <= int(width * 0.24)
        and line["width"] <= int(width * 0.22)
        and len(_normalize_text(line["text"], limit=80)) <= 24
        and len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", line["text"])) <= 3
    ]
    chat_title_candidates = [
        line
        for line in top_lines
        if not _looks_like_url_or_query(line["text"])
        and len(line["text"]) <= 40
        and int(width * 0.25) <= line["x"] <= int(width * 0.78)
        and len(re.findall(r"[A-Za-z\u4e00-\u9fff]", line["text"])) >= 2
    ]
    chat_keyword_hits, _ = _count_keyword_hits(lowered_text, _CHAT_KEYWORDS)
    left_message_lines = [line for line in left_lines if len(line["text"]) >= 8]
    right_message_lines = [line for line in right_lines if len(line["text"]) >= 6]
    if (
        input_candidates
        and (left_message_lines or right_message_lines or chat_keyword_hits >= 1 or len(sidebar_lines) >= 2)
    ) or (
        left_message_lines
        and right_message_lines
        and (chat_keyword_hits >= 1 or len(sidebar_lines) >= 1)
    ):
        title = (
            min(chat_title_candidates, key=lambda line: (line["y"], -line["width"]))["text"]
            if chat_title_candidates
            else ""
        )
        return {
            "kind": "chat",
            "title": _repair_short_ui_text(title),
            "left_messages": _dedupe_texts([_stabilize_layout_text(line["text"]) for line in left_message_lines[:4]], limit=4),
            "right_messages": _dedupe_texts([_stabilize_layout_text(line["text"]) for line in right_message_lines[:4]], limit=4),
            "input_hint": _repair_short_ui_text(input_candidates[0]["text"]) if input_candidates else "",
            "sidebar_labels": _normalize_short_text_list([line["text"] for line in sidebar_lines[:4]], limit=4),
        }
    return {}


def _format_layout_details(layout: dict) -> str:
    if not layout:
        return ""
    kind = str(layout.get("kind") or "").strip().lower()
    if kind == "browser":
        parts: list[str] = []
        if layout.get("page_title"):
            parts.append(f"page title: {_stabilize_layout_text(layout['page_title'])}")
        if layout.get("address_bar"):
            parts.append(f"address bar: {_stabilize_layout_text(layout['address_bar'])}")
        field_labels = [str(item).strip() for item in layout.get("field_labels") or [] if str(item).strip()]
        if field_labels:
            parts.append(f"visible controls: {', '.join(field_labels[:4])}")
        return "; ".join(parts)
    if kind == "chat":
        parts = []
        if layout.get("title"):
            parts.append(f"chat title: {_stabilize_layout_text(layout['title'])}")
        if layout.get("left_messages") and layout.get("right_messages"):
            parts.append("messages appear on both left and right sides")
        elif layout.get("left_messages"):
            parts.append("messages are visible on the left side")
        if layout.get("input_hint"):
            parts.append(f"composer: {_stabilize_layout_text(layout['input_hint'])}")
        sidebar_labels = [str(item).strip() for item in layout.get("sidebar_labels") or [] if str(item).strip()]
        if sidebar_labels:
            parts.append(f"sidebar items: {', '.join(sidebar_labels[:3])}")
        return "; ".join(parts)
    return ""


def _format_chart_text_details(chart_text: dict) -> str:
    if not chart_text:
        return ""
    parts: list[str] = []
    title = str(chart_text.get("title") or "").strip()
    if title:
        parts.append(f"title: {title}")
    x_axis = [str(item).strip() for item in chart_text.get("x_axis_labels") or [] if str(item).strip()]
    y_axis = [str(item).strip() for item in chart_text.get("y_axis_labels") or [] if str(item).strip()]
    legend = [str(item).strip() for item in chart_text.get("legend_labels") or [] if str(item).strip()]
    if x_axis:
        parts.append(f"x-axis: {', '.join(x_axis[:6])}")
    if y_axis:
        parts.append(f"y-axis: {', '.join(y_axis[:6])}")
    if legend:
        parts.append(f"legend/annotations: {', '.join(legend[:4])}")
    return "; ".join(parts)


def _collect_layout_visible_text(layout: dict) -> list[str]:
    if not layout:
        return []
    kind = str(layout.get("kind") or "").strip().lower()
    if kind == "browser":
        return _dedupe_texts(
            ([layout.get("page_title")] if layout.get("page_title") else [])
            + ([layout.get("address_bar")] if layout.get("address_bar") else [])
            + list(layout.get("field_labels") or []),
            limit=10,
        )
    if kind == "chat":
        return _dedupe_texts(
            ([layout.get("title")] if layout.get("title") else [])
            + list(layout.get("sidebar_labels") or [])
            + list(layout.get("left_messages") or [])
            + list(layout.get("right_messages") or [])
            + ([layout.get("input_hint")] if layout.get("input_hint") else []),
            limit=12,
        )
    return []


def _collect_chart_visible_text(chart_text: dict, ocr_text: str) -> list[str]:
    if not chart_text:
        return []
    return _dedupe_texts(
        ([chart_text.get("title")] if chart_text.get("title") else [])
        + list(chart_text.get("x_axis_labels") or [])
        + list(chart_text.get("y_axis_labels") or [])
        + list(chart_text.get("legend_labels") or [])
        + _extract_numeric_markers(ocr_text, limit=6),
        limit=12,
    )


def _build_clean_visible_text(*, scene: str, layout: dict, chart_text: dict, ocr_text: str) -> str:
    scene = str(scene or "").strip().lower()
    structured = _dedupe_texts(
        _collect_layout_visible_text(layout) + _collect_chart_visible_text(chart_text, ocr_text),
        limit=12,
    )
    if structured and scene in {"browser", "chat", "chart"}:
        return _normalize_text("; ".join(structured), limit=1200)
    if structured:
        extra = _extract_readable_labels(ocr_text, limit=6)
        return _normalize_text("; ".join(_dedupe_texts(structured + extra, limit=12)), limit=1200)
    return _normalize_text(ocr_text, limit=1200)


def _structured_summary_labels(scene: str, layout: dict, chart_text: dict, fallback: list[str]) -> list[str]:
    scene = str(scene or "").strip().lower()
    if scene == "browser":
        labels = _dedupe_texts(
            list(layout.get("field_labels") or [])
            + ([layout.get("page_title")] if layout.get("page_title") else []),
            limit=6,
        )
        return labels or fallback
    if scene == "chat":
        labels = _dedupe_texts(
            list(layout.get("sidebar_labels") or [])
            + ([layout.get("input_hint")] if layout.get("input_hint") else [])
            + ([layout.get("title")] if layout.get("title") else []),
            limit=6,
        )
        return labels or fallback
    if scene == "chart":
        labels = _dedupe_texts(
            ([chart_text.get("title")] if chart_text.get("title") else [])
            + list(chart_text.get("x_axis_labels") or [])
            + list(chart_text.get("legend_labels") or []),
            limit=6,
        )
        return labels or fallback
    return fallback


def _collect_active_segments(values, *, threshold: int, min_width: int) -> list[tuple[int, int]]:
    active = [int(value) > threshold for value in values]
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_active in enumerate(active):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            if index - start >= min_width:
                segments.append((start, index - 1))
            start = None
    if start is not None and len(active) - start >= min_width:
        segments.append((start, len(active) - 1))
    return segments


def _detect_direction_from_series(values: list[float], *, threshold: float) -> str:
    filtered = [float(value) for value in values if value is not None]
    if len(filtered) < 2:
        return ""
    delta = filtered[-1] - filtered[0]
    if abs(delta) < threshold:
        return ""
    positive_steps = sum(1 for left, right in zip(filtered, filtered[1:]) if right - left > 0)
    negative_steps = sum(1 for left, right in zip(filtered, filtered[1:]) if right - left < 0)
    if delta < 0 and negative_steps >= positive_steps:
        return "upward"
    if delta > 0 and positive_steps >= negative_steps:
        return "downward"
    return ""


def _analyze_chart_visual_pattern(image: Image.Image) -> dict:
    try:
        import numpy as np
    except Exception:
        return {}

    try:
        rgb = image.convert("RGB")
        max_width = 960
        if rgb.width > max_width:
            ratio = max_width / float(max(rgb.width, 1))
            rgb = rgb.resize(
                (
                    max(1, int(round(rgb.width * ratio))),
                    max(1, int(round(rgb.height * ratio))),
                ),
                _RESAMPLING.BILINEAR,
            )
        arr = np.asarray(rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return {}
        channel_span = arr.max(axis=2) - arr.min(axis=2)
        brightness = arr.mean(axis=2)
        accent_mask = (channel_span > 40) & (brightness > 35) & (brightness < 245)
        ys, xs = np.nonzero(accent_mask)
        if len(xs) < 200:
            return {}

        bbox_x0 = int(xs.min())
        bbox_x1 = int(xs.max())
        bbox_y0 = int(ys.min())
        bbox_y1 = int(ys.max())
        bbox_w = bbox_x1 - bbox_x0 + 1
        bbox_h = bbox_y1 - bbox_y0 + 1
        if bbox_w < int(arr.shape[1] * 0.22) or bbox_h < int(arr.shape[0] * 0.12):
            return {}

        sub_mask = accent_mask[bbox_y0 : bbox_y1 + 1, bbox_x0 : bbox_x1 + 1]
        col_counts = sub_mask.sum(axis=0)
        low_segments = _collect_active_segments(
            col_counts,
            threshold=max(2, int(bbox_h * 0.012)),
            min_width=max(10, int(bbox_w * 0.04)),
        )
        high_segments = _collect_active_segments(
            col_counts,
            threshold=max(8, int(bbox_h * 0.05)),
            min_width=max(12, int(bbox_w * 0.04)),
        )

        bar_components: list[dict] = []
        for seg_x0, seg_x1 in high_segments:
            region = sub_mask[:, seg_x0 : seg_x1 + 1]
            seg_ys, _seg_xs = np.nonzero(region)
            if len(seg_ys) == 0:
                continue
            top = int(seg_ys.min())
            bottom = int(seg_ys.max())
            width = int(seg_x1 - seg_x0 + 1)
            height = int(bottom - top + 1)
            fill = float(len(seg_ys) / max(width * height, 1))
            if height < int(bbox_h * 0.18):
                continue
            bar_components.append(
                {
                    "x0": int(seg_x0),
                    "x1": int(seg_x1),
                    "top": top,
                    "bottom": bottom,
                    "width": width,
                    "height": height,
                    "fill": fill,
                }
            )

        if len(bar_components) >= 3:
            bottoms = [comp["bottom"] for comp in bar_components]
            if (
                max(bottoms) - min(bottoms) <= max(12, int(bbox_h * 0.12))
                and max(comp["width"] for comp in bar_components) <= int(bbox_w * 0.28)
                and min(comp["fill"] for comp in bar_components) >= 0.55
            ):
                heights = [-float(comp["height"]) for comp in bar_components]
                trend = _detect_direction_from_series(heights, threshold=max(16.0, bbox_h * 0.12))
                return {
                    "chart_like": True,
                    "kind": "bar",
                    "trend": trend,
                    "series_count": len(bar_components),
                    "bbox": {"width": int(bbox_w), "height": int(bbox_h)},
                }

        if len(low_segments) == 1:
            seg_x0, seg_x1 = low_segments[0]
            region = sub_mask[:, seg_x0 : seg_x1 + 1]
            bins = np.array_split(np.arange(region.shape[1]), 8)
            centroids: list[float] = []
            for xbin in bins:
                sample = region[:, xbin]
                bin_ys, _bin_xs = np.nonzero(sample)
                if len(bin_ys) < max(10, int(bbox_h * 0.01)):
                    continue
                centroids.append(float(bin_ys.mean()))
            if len(centroids) >= 4:
                trend = _detect_direction_from_series(centroids, threshold=max(14.0, bbox_h * 0.12))
                if trend:
                    return {
                        "chart_like": True,
                        "kind": "line",
                        "trend": trend,
                        "series_count": 1,
                        "bbox": {"width": int(bbox_w), "height": int(bbox_h)},
                    }
        return {}
    except Exception:
        return {}


def _detect_visual_scene(
    image: Image.Image,
    *,
    caption: str,
    ocr_text: str,
    chart_visual: dict | None = None,
) -> str:
    combined = " ".join(part for part in (caption, ocr_text) if str(part or "").strip())
    lowered = _normalize_text(combined, limit=1600).lower()
    chart_visual = chart_visual or {}
    if not lowered:
        if chart_visual.get("chart_like"):
            return "chart"
        return "ui" if image.width >= 900 and image.height >= 500 else "image"

    settings_count, _ = _count_keyword_hits(lowered, _SETTINGS_KEYWORDS)
    chart_count, _ = _count_keyword_hits(lowered, _CHART_KEYWORDS)
    browser_count, _ = _count_keyword_hits(lowered, _BROWSER_KEYWORDS)
    chat_count, _ = _count_keyword_hits(lowered, _CHAT_KEYWORDS)
    numeric_markers = _extract_numeric_markers(lowered, limit=8)
    q_markers = [marker for marker in numeric_markers if marker.lower().startswith("q")]

    if chart_count >= 1 and (len(numeric_markers) >= 3 or bool(q_markers) or "dashboard" in lowered):
        return "chart"
    if chart_visual.get("chart_like"):
        return "chart"
    if settings_count >= 2:
        return "settings"
    if browser_count >= 2:
        return "browser"
    if chat_count >= 2:
        return "chat"

    word_count = len(re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", lowered))
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", lowered))
    if image.width >= 900 and image.height >= 500 and (word_count >= 18 or cjk_count >= 32):
        return "document"
    if image.width >= 900 and image.height >= 500:
        return "ui"
    return "image"


def _join_human_list(items: list[str]) -> str:
    values = [str(item or "").strip() for item in items if str(item or "").strip()]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _infer_settings_focus(labels: list[str], ocr_text: str) -> list[str]:
    lowered = _normalize_text(" ".join(labels + [ocr_text]), limit=1200).lower()
    focus: list[str] = []
    if any(keyword in lowered for keyword in ("battery", "power", "brightness", "display", "screen")):
        focus.append("battery and display")
    if any(keyword in lowered for keyword in ("network", "bluetooth", "wifi", "wi-fi")):
        focus.append("connectivity")
    if any(keyword in lowered for keyword in ("privacy", "security", "permission")):
        focus.append("privacy")
    if any(keyword in lowered for keyword in ("volume", "audio", "sound")):
        focus.append("audio")
    return focus[:3]


def _choose_caption_prompt(image: Image.Image, ocr_text: str) -> str:
    lowered = _normalize_text(ocr_text, limit=600).lower()
    if any(keyword in lowered for keyword in _SETTINGS_KEYWORDS):
        return "a computer settings screen showing"
    if any(keyword in lowered for keyword in _CHART_KEYWORDS):
        return "a dashboard or chart showing"
    if any(keyword in lowered for keyword in _BROWSER_KEYWORDS):
        return "a browser window showing"
    if any(keyword in lowered for keyword in _CHAT_KEYWORDS):
        return "a chat window showing"
    if image.width >= 900 and image.height >= 500:
        return "a desktop app window showing"
    return "a screenshot of"


def _should_attempt_caption(image: Image.Image, ocr_text: str) -> bool:
    normalized = _normalize_text(ocr_text, limit=1200)
    if not normalized:
        return True
    if _is_low_signal_ocr(normalized):
        return True
    chart_visual = _analyze_chart_visual_pattern(image)
    scene = _detect_visual_scene(image, caption="", ocr_text=normalized, chart_visual=chart_visual)
    if scene in {"chart", "browser", "chat"}:
        return True
    latin_words = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", normalized)
    return (
        image.width >= 900
        and image.height >= 500
        and len(normalized) < 64
        and len(latin_words) <= 7
    )


def _build_visual_summary(
    *,
    image: Image.Image,
    caption: str,
    ocr_text: str,
    scene: str | None = None,
    chart_visual: dict | None = None,
    chart_text: dict | None = None,
    layout: dict | None = None,
) -> str:
    caption_text = _normalize_text(caption, limit=320)
    labels = _extract_readable_labels(ocr_text, limit=6)
    numeric_markers = _extract_numeric_markers(ocr_text, limit=8)
    chart_visual = chart_visual or {}
    chart_text = chart_text or {}
    layout = layout or {}
    scene = str(scene or _detect_visual_scene(image, caption=caption_text, ocr_text=ocr_text, chart_visual=chart_visual))
    labels = _structured_summary_labels(scene, layout, chart_text, labels)

    if scene == "settings":
        focus = _infer_settings_focus(labels, ocr_text)
        base = "This appears to be a system settings screen"
        if focus:
            base += f" focused on {_join_human_list(focus)} options."
        else:
            base += "."
        if labels:
            base += f" Readable labels include: {', '.join(labels)}."
        return base

    if scene == "chart":
        kind = str(chart_visual.get("kind") or "").strip().lower()
        trend = str(chart_visual.get("trend") or "").strip().lower()
        if kind == "line":
            base = "This looks like a dashboard featuring a line chart"
        elif kind == "bar":
            base = "This looks like a dashboard featuring a bar chart"
        else:
            base = "This looks like a dashboard or chart with numeric metrics"
        if trend == "upward":
            base += " with an upward left-to-right trend"
        elif trend == "downward":
            base += " with a downward left-to-right trend"
        if any(marker.lower().startswith("q") for marker in numeric_markers):
            base += " arranged by quarter"
        if any("%" in marker for marker in numeric_markers):
            base += " and a visible growth indicator"
        base += "."
        detail_parts: list[str] = []
        chart_detail_text = _format_chart_text_details(chart_text)
        if chart_detail_text:
            detail_parts.append(f"Chart text details: {chart_detail_text}.")
        if labels:
            detail_parts.append(f"Readable labels include: {', '.join(labels)}.")
        if numeric_markers:
            detail_parts.append(f"Visible markers include: {', '.join(numeric_markers[:6])}.")
        if caption_text:
            detail_parts.append(f"Caption hint: {caption_text}.")
        return " ".join([base] + detail_parts).strip()

    if scene == "browser":
        base = "This appears to be a browser or website page."
        layout_text = _format_layout_details(layout)
        if layout_text:
            base += f" Layout details: {layout_text}."
        if labels:
            base += f" Readable labels include: {', '.join(labels)}."
        return base

    if scene == "chat":
        base = "This appears to be a chat or messaging interface."
        layout_text = _format_layout_details(layout)
        if layout_text:
            base += f" Layout details: {layout_text}."
        if labels:
            base += f" Readable labels include: {', '.join(labels)}."
        return base

    if scene == "document":
        base = "This appears to be a text-heavy document or page."
        if labels:
            base += f" Readable labels include: {', '.join(labels)}."
        return base

    if caption_text and labels:
        return f"{caption_text}. Readable labels include: {', '.join(labels)}."
    if caption_text and ocr_text:
        return f"{caption_text}. The image also contains readable text."
    if caption_text:
        return caption_text
    if labels:
        kind = "interface or screenshot" if image.width >= 900 and image.height >= 500 else "image"
        return f"The {kind} includes readable labels such as {', '.join(labels)}."
    if ocr_text:
        return "The image contains readable text."
    return "Unable to extract stable detail from the image."


def _get_caption_backend_state() -> dict[str, bool]:
    with _CAPTION_LOCK:
        return {
            "ready": _CAPTION_BACKEND is not None,
            "loading": _CAPTION_LOADING,
            "error": bool(_CAPTION_LOAD_ERROR),
        }


def _run_windows_ocr(image: Image.Image, debug_write: _DEBUG_WRITE, *, include_layout: bool = False) -> str | dict:
    if os.name != "nt":
        return ""
    powershell = shutil.which("powershell")
    if not powershell:
        return ""
    tmp_path = ""
    timeout_seconds = _ocr_timeout_seconds()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as handle:
            tmp_path = handle.name
        image.save(tmp_path, format="PNG", optimize=True)
        if include_layout:
            script = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
  $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 -and $_.GetGenericArguments().Count -eq 1
} | Select-Object -First 1)
function Await($op, [Type]$resultType) {
  $task = $asTaskGeneric.MakeGenericMethod(@($resultType)).Invoke($null, @($op))
  $null = $task.Wait(-1)
  return $task.Result
}
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Storage.FileAccessMode, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.IRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrResult, Windows.Media.Ocr, ContentType = WindowsRuntime]
$null = [Windows.Globalization.Language, Windows.Globalization, ContentType = WindowsRuntime]
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$path = '__IMAGE_PATH__'
$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($path)) ([Windows.Storage.StorageFile])
$stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if ($null -eq $engine) {
  $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage((New-Object Windows.Globalization.Language('zh-CN')))
}
if ($null -eq $engine) {
  $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage((New-Object Windows.Globalization.Language('en-US')))
}
if ($null -eq $engine) {
  [ordered]@{ text=''; lines=@() } | ConvertTo-Json -Compress -Depth 5
  exit 0
}
$result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
$lines = @()
foreach($line in $result.Lines) {
  $boxes = @()
  foreach($word in $line.Words) {
    try {
      $rect = $word.BoundingRect
      if ($null -ne $rect) { $boxes += $rect }
    } catch {}
  }
  if ($boxes.Count -gt 0) {
    $minX = ($boxes | Measure-Object X -Minimum).Minimum
    $minY = ($boxes | Measure-Object Y -Minimum).Minimum
    $maxR = ($boxes | ForEach-Object { $_.X + $_.Width } | Measure-Object -Maximum).Maximum
    $maxB = ($boxes | ForEach-Object { $_.Y + $_.Height } | Measure-Object -Maximum).Maximum
    $width = [int]([Math]::Max(0, $maxR - $minX))
    $height = [int]([Math]::Max(0, $maxB - $minY))
  } else {
    $minX = 0
    $minY = 0
    $width = 0
    $height = 0
  }
  $lines += [ordered]@{
    text = $line.Text
    x = [int]$minX
    y = [int]$minY
    width = $width
    height = $height
  }
}
[ordered]@{
  text = if ($null -ne $result -and $null -ne $result.Text) { $result.Text } else { '' }
  lines = $lines
} | ConvertTo-Json -Compress -Depth 6
"""
        else:
            script = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
  $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 -and $_.GetGenericArguments().Count -eq 1
} | Select-Object -First 1)
function Await($op, [Type]$resultType) {
  $task = $asTaskGeneric.MakeGenericMethod(@($resultType)).Invoke($null, @($op))
  $null = $task.Wait(-1)
  return $task.Result
}
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Storage.FileAccessMode, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.IRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrResult, Windows.Media.Ocr, ContentType = WindowsRuntime]
$null = [Windows.Globalization.Language, Windows.Globalization, ContentType = WindowsRuntime]
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$path = '__IMAGE_PATH__'
$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($path)) ([Windows.Storage.StorageFile])
$stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if ($null -eq $engine) {
  $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage((New-Object Windows.Globalization.Language('zh-CN')))
}
if ($null -eq $engine) {
  $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage((New-Object Windows.Globalization.Language('en-US')))
}
if ($null -eq $engine) { exit 0 }
$result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
if ($null -ne $result -and $null -ne $result.Text) {
  Write-Output $result.Text
}
"""
        script = script.replace("__IMAGE_PATH__", tmp_path.replace("'", "''"))
        completed = subprocess.run(
            [powershell, "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            stderr = _normalize_text(completed.stderr, limit=200)
            if stderr:
                debug_write("vision_local_ocr_error", {"error": stderr})
            return {"text": "", "lines": []} if include_layout else ""
        stdout = str(completed.stdout or "")
        if include_layout:
            try:
                payload = json.loads(stdout)
                text = _normalize_text(payload.get("text", ""), limit=1600)
                lines = payload.get("lines", [])
                return {"text": text, "lines": lines if isinstance(lines, list) else []}
            except Exception:
                return {"text": _normalize_text(stdout, limit=1600), "lines": []}
        return _normalize_text(stdout, limit=1600)
    except subprocess.TimeoutExpired:
        debug_write("vision_local_ocr_timeout", {"timeout_s": timeout_seconds})
        return {"text": "", "lines": []} if include_layout else ""
    except Exception as exc:
        debug_write("vision_local_ocr_exception", {"error": str(exc)})
        return {"text": "", "lines": []} if include_layout else ""
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _ensure_caption_backend_loading(debug_write: _DEBUG_WRITE) -> None:
    global _CAPTION_LOADING
    if _CAPTION_BACKEND is not None:
        return
    if _CAPTION_LOAD_ATTEMPTED and _CAPTION_LOAD_ERROR:
        return
    with _CAPTION_LOCK:
        if _CAPTION_BACKEND is not None:
            return
        if _CAPTION_LOAD_ATTEMPTED and _CAPTION_LOAD_ERROR:
            return
        if _CAPTION_LOADING:
            return
        _CAPTION_LOADING = True

    def _runner() -> None:
        global _CAPTION_LOADING
        try:
            _load_caption_backend(debug_write, blocking=True)
        finally:
            with _CAPTION_LOCK:
                _CAPTION_LOADING = False

    thread = threading.Thread(target=_runner, name="vision-caption-loader", daemon=True)
    thread.start()
    debug_write("vision_local_caption_loading", {"background": True})


def _load_caption_backend(debug_write: _DEBUG_WRITE, *, blocking: bool = False) -> tuple[object, object] | None:
    global _CAPTION_BACKEND, _CAPTION_LOAD_ATTEMPTED, _CAPTION_LOAD_ERROR
    if _CAPTION_BACKEND is not None:
        return _CAPTION_BACKEND
    if _CAPTION_LOAD_ATTEMPTED and _CAPTION_LOAD_ERROR:
        return None
    if not blocking:
        _ensure_caption_backend_loading(debug_write)
        return _CAPTION_BACKEND

    with _CAPTION_LOCK:
        if _CAPTION_BACKEND is not None:
            return _CAPTION_BACKEND
        if _CAPTION_LOAD_ATTEMPTED and _CAPTION_LOAD_ERROR:
            return None
        _CAPTION_LOAD_ATTEMPTED = True

    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor
        from transformers.utils import logging as transformers_logging

        model_id = (
            os.environ.get("AARONCORE_LOCAL_CAPTION_MODEL")
            or "Salesforce/blip-image-captioning-base"
        ).strip()
        local_only = not _caption_allow_download()
        transformers_logging.set_verbosity_error()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            processor = BlipProcessor.from_pretrained(model_id, local_files_only=local_only)
            model = BlipForConditionalGeneration.from_pretrained(model_id, local_files_only=local_only)
        model.eval()
        with _CAPTION_LOCK:
            _CAPTION_BACKEND = (processor, model)
            _CAPTION_LOAD_ERROR = ""
        debug_write(
            "vision_local_caption_ready",
            {"model_id": model_id, "local_only": local_only},
        )
    except Exception as exc:
        with _CAPTION_LOCK:
            _CAPTION_BACKEND = None
            _CAPTION_LOAD_ERROR = str(exc)
        debug_write("vision_local_caption_unavailable", {"error": str(exc)[:240]})
    return _CAPTION_BACKEND


def _caption_image(image: Image.Image, debug_write: _DEBUG_WRITE, *, prompt: str = "") -> str:
    backend = _load_caption_backend(debug_write, blocking=_caption_blocking_enabled())
    if backend is None:
        return ""
    processor, model = backend
    try:
        import torch

        kwargs = {"images": image, "return_tensors": "pt"}
        if str(prompt or "").strip():
            kwargs["text"] = str(prompt).strip()
        inputs = processor(**kwargs)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=64)
        if output is None or len(output) == 0:
            return ""
        return _normalize_text(processor.decode(output[0], skip_special_tokens=True), limit=320)
    except Exception as exc:
        debug_write("vision_local_caption_error", {"error": str(exc)[:240]})
        return ""


def has_local_image_support() -> bool:
    if os.name == "nt" and shutil.which("powershell"):
        return True
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def analyze_image(image_b64: str, *, debug_write: _DEBUG_WRITE | None = None) -> dict:
    debug_write = debug_write or _noop_debug
    started_at = time.perf_counter()
    image, raw = _decode_image(image_b64)
    if image is None or not raw:
        return {
            "ok": False,
            "caption": "",
            "ocr_text": "",
            "summary": "Unable to decode the uploaded image.",
            "digest": "",
            "size": "",
        }

    digest = hashlib.sha256(raw).hexdigest()
    cached = _cache_get(digest)
    if cached is not None:
        return cached

    if image.width >= 900 and image.height >= 500:
        _load_caption_backend(debug_write, blocking=False)

    ocr_started_at = time.perf_counter()
    ocr_payload = _run_windows_ocr(image, debug_write, include_layout=True)
    if isinstance(ocr_payload, dict):
        ocr_text = _normalize_text(ocr_payload.get("text", ""), limit=1600)
        ocr_lines = _normalize_ocr_lines(image, ocr_payload.get("lines", []))
    else:
        ocr_text = _normalize_text(ocr_payload, limit=1600)
        ocr_lines = []
    ocr_retried = False
    if _should_retry_ocr(image, ocr_text):
        retry_image = _build_ocr_retry_image(image)
        retry_payload = _run_windows_ocr(retry_image, debug_write, include_layout=True)
        ocr_retried = True
        if isinstance(retry_payload, dict):
            retry_text = _normalize_text(retry_payload.get("text", ""), limit=1600)
            retry_lines = _normalize_ocr_lines(
                image,
                retry_payload.get("lines", []),
                source_size=(retry_image.width, retry_image.height),
            )
        else:
            retry_text = _normalize_text(retry_payload, limit=1600)
            retry_lines = []
        if retry_lines:
            ocr_lines = _merge_ocr_lines(ocr_lines, retry_lines)
        if _ocr_signal_score(retry_text) > _ocr_signal_score(ocr_text):
            ocr_text = retry_text
    ocr_ms = round((time.perf_counter() - ocr_started_at) * 1000, 1)
    chart_visual = _analyze_chart_visual_pattern(image)
    pre_scene = _detect_visual_scene(image, caption="", ocr_text=ocr_text, chart_visual=chart_visual)
    layout = _analyze_structured_layout(image, ocr_lines, ocr_text)
    chart_text = _extract_chart_text_structure(image, ocr_lines) if ocr_lines and pre_scene == "chart" else {}

    caption = ""
    caption_ms = 0.0
    caption_pending = False
    if _should_attempt_caption(image, ocr_text):
        caption_started_at = time.perf_counter()
        caption = _caption_image(
            image,
            debug_write,
            prompt=_choose_caption_prompt(image, ocr_text),
        )
        caption_ms = round((time.perf_counter() - caption_started_at) * 1000, 1)
        state = _get_caption_backend_state()
        caption_pending = not caption and state["loading"] and not state["error"]

    size = f"{image.width}x{image.height}"
    scene = _detect_visual_scene(
        image,
        caption=caption,
        ocr_text=ocr_text,
        chart_visual=chart_visual,
    )
    layout_kind = str(layout.get("kind") or "").strip().lower()
    if scene in {"ui", "document"} and layout_kind in {"browser", "chat"}:
        scene = layout_kind
    if scene == "chart" and not chart_text:
        chart_text = _extract_chart_text_structure(image, ocr_lines)
    visible_text = _build_clean_visible_text(scene=scene, layout=layout, chart_text=chart_text, ocr_text=ocr_text)

    analysis = {
        "ok": True,
        "caption": caption,
        "ocr_text": ocr_text,
        "visible_text": visible_text,
        "summary": _build_visual_summary(
            image=image,
            caption=caption,
            ocr_text=ocr_text,
            scene=scene,
            chart_visual=chart_visual,
            chart_text=chart_text,
            layout=layout,
        ),
        "scene": scene,
        "chart_visual": dict(chart_visual) if chart_visual else {},
        "chart_text": dict(chart_text) if chart_text else {},
        "layout": dict(layout) if layout else {},
        "digest": digest,
        "size": size,
        "caption_pending": caption_pending,
    }
    if not caption_pending:
        _cache_put(digest, analysis)
    debug_write(
        "vision_local_analysis_timing",
        {
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 1),
            "ocr_ms": ocr_ms,
            "caption_ms": caption_ms,
            "has_ocr": bool(ocr_text),
            "has_caption": bool(caption),
            "ocr_retried": ocr_retried,
            "caption_pending": caption_pending,
            "size": size,
        },
    )
    return dict(analysis)


def build_user_image_context(
    images: list[str] | None,
    *,
    user_text: str = "",
    debug_write: _DEBUG_WRITE | None = None,
) -> str:
    debug_write = debug_write or _noop_debug
    image_list = [str(item or "").strip() for item in (images or []) if str(item or "").strip()]
    if not image_list:
        return ""

    lines = [
        "[LOCAL_IMAGE_CONTEXT]",
        "Below is locally extracted visual context from the user's uploaded images.",
        "Answer directly from this information.",
        "Do not mention OCR, caption models, local analysis, model switching, or any claim that you cannot see images.",
    ]
    if str(user_text or "").strip():
        lines.append(f"User question: {_normalize_text(user_text, limit=400)}")

    for index, image_b64 in enumerate(image_list[:4], start=1):
        analysis = analyze_image(image_b64, debug_write=debug_write)
        lines.append(f"Image {index}:")
        scene = str(analysis.get("scene") or "").strip()
        if scene:
            lines.append(f"- Scene type: {scene}")
        lines.append(
            f"- Visual summary: {analysis.get('summary') or 'Unable to extract stable detail from the image.'}"
        )
        layout_details = _format_layout_details(analysis.get("layout") or {})
        if layout_details:
            lines.append(f"- Layout details: {layout_details}")
        chart_details = _format_chart_text_details(analysis.get("chart_text") or {})
        if chart_details:
            lines.append(f"- Chart details: {chart_details}")
        visible_text = _normalize_text(analysis.get("visible_text", ""), limit=1200)
        if visible_text:
            lines.append(f"- Visible text: {visible_text}")
        size = str(analysis.get("size") or "").strip()
        if size:
            lines.append(f"- Image size: {size}")

    omitted = len(image_list) - min(len(image_list), 4)
    if omitted > 0:
        lines.append(f"{omitted} additional image(s) were omitted from expansion.")
    return "\n".join(lines)


def build_screen_description(image_b64: str, *, debug_write: _DEBUG_WRITE | None = None) -> str:
    debug_write = debug_write or _noop_debug
    analysis = analyze_image(image_b64, debug_write=debug_write)
    caption = _normalize_text(analysis.get("caption", ""), limit=180)
    summary = _normalize_text(analysis.get("summary", ""), limit=180)
    has_ocr = bool(_normalize_text(analysis.get("ocr_text", ""), limit=80))
    if caption and has_ocr:
        return f"{caption}. The screen also contains a substantial amount of readable text."
    if caption:
        return caption
    if summary and summary not in {
        "The image contains readable text.",
        "Unable to extract stable detail from the image.",
    }:
        return summary
    if has_ocr:
        return "The screen appears to be a text-heavy interface."
    return summary
