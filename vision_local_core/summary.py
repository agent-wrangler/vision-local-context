from __future__ import annotations

import re

from PIL import Image

from . import shared as _shared
from .layout import (
    _analyze_chart_visual_pattern,
    _detect_visual_scene,
    _format_chart_text_details,
    _format_layout_details,
    _structured_summary_labels,
)

_BROWSER_KEYWORDS = _shared._BROWSER_KEYWORDS
_CHART_KEYWORDS = _shared._CHART_KEYWORDS
_CHAT_KEYWORDS = _shared._CHAT_KEYWORDS
_SETTINGS_KEYWORDS = _shared._SETTINGS_KEYWORDS
_extract_numeric_markers = _shared._extract_numeric_markers
_extract_readable_labels = _shared._extract_readable_labels
_is_low_signal_ocr = _shared._is_low_signal_ocr
_normalize_text = _shared._normalize_text


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
    if any(
        keyword in lowered
        for keyword in (
            "battery",
            "power",
            "brightness",
            "display",
            "screen",
            "\u7535\u6c60",
            "\u4eae\u5ea6",
            "\u663e\u793a",
            "\u5c4f\u5e55",
        )
    ):
        focus.append("battery and display")
    if any(
        keyword in lowered
        for keyword in (
            "network",
            "bluetooth",
            "wifi",
            "wi-fi",
            "\u7f51\u7edc",
            "\u84dd\u7259",
            "\u65e0\u7ebf",
        )
    ):
        focus.append("connectivity")
    if any(
        keyword in lowered
        for keyword in (
            "privacy",
            "security",
            "permission",
            "\u9690\u79c1",
            "\u6743\u9650",
            "\u5b89\u5168",
        )
    ):
        focus.append("privacy")
    if any(
        keyword in lowered
        for keyword in (
            "volume",
            "audio",
            "sound",
            "\u97f3\u91cf",
            "\u58f0\u97f3",
            "\u97f3\u9891",
        )
    ):
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


__all__ = [
    "_build_visual_summary",
    "_choose_caption_prompt",
    "_extract_numeric_markers",
    "_extract_readable_labels",
    "_infer_settings_focus",
    "_join_human_list",
    "_normalize_text",
    "_should_attempt_caption",
]
