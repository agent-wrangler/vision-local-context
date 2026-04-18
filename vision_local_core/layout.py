from __future__ import annotations

import difflib
import re

from PIL import Image

from . import shared as _shared

_BROWSER_FIELD_TOKENS = _shared._BROWSER_FIELD_TOKENS
_BROWSER_KEYWORDS = _shared._BROWSER_KEYWORDS
_BROWSER_TITLE_EXCLUDE_TOKENS = _shared._BROWSER_TITLE_EXCLUDE_TOKENS
_CHAT_INPUT_TOKENS = _shared._CHAT_INPUT_TOKENS
_CHAT_KEYWORDS = _shared._CHAT_KEYWORDS
_CHART_KEYWORDS = _shared._CHART_KEYWORDS
_COMMON_UI_CANONICALS = _shared._COMMON_UI_CANONICALS
_RESAMPLING = _shared._RESAMPLING
_SETTINGS_KEYWORDS = _shared._SETTINGS_KEYWORDS
_extract_numeric_markers = _shared._extract_numeric_markers
_extract_readable_labels = _shared._extract_readable_labels
_looks_like_url_or_query = _shared._looks_like_url_or_query
_normalize_text = _shared._normalize_text
_ocr_line_text_quality = _shared._ocr_line_text_quality


def _count_keyword_hits(text: str, keywords: set[str]) -> tuple[int, set[str]]:
    lowered = _normalize_text(text, limit=1600).lower()
    hits = {keyword for keyword in keywords if keyword in lowered}
    return len(hits), hits


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
    if re.search(r"[\u4e00-\u9fff]", normalized) and re.search(r"[A-Za-z]", normalized):
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
    body_lines = [line for line in ocr_lines if int(height * 0.10) <= line["y"] <= int(height * 0.84)]
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
        and not any(token in line["text"].lower() for token in _BROWSER_TITLE_EXCLUDE_TOKENS)
    ]
    field_candidates = [line for line in body_lines if any(token in line["text"].lower() for token in _BROWSER_FIELD_TOKENS)]
    if address_candidates or ("browser" in lowered_text and field_candidates):
        address_text = max(address_candidates, key=lambda line: line["width"])["text"] if address_candidates else ""
        page_title = (
            max(browser_title_candidates, key=lambda line: (len(line["text"]), line["width"]))["text"]
            if browser_title_candidates
            else ""
        )
        field_label_candidates = [
            line
            for line in field_candidates
            if _normalize_text(line.get("text", ""), limit=120) != page_title
        ]
        field_labels = _normalize_short_text_list([line["text"] for line in field_label_candidates[:4]], limit=4)
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
        and any(token in line["text"].lower() for token in _CHAT_INPUT_TOKENS)
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


__all__ = [
    "_ambiguous_signature",
    "_analyze_chart_visual_pattern",
    "_analyze_structured_layout",
    "_build_clean_visible_text",
    "_collect_active_segments",
    "_collect_chart_visible_text",
    "_collect_layout_visible_text",
    "_count_keyword_hits",
    "_dedupe_texts",
    "_detect_direction_from_series",
    "_detect_visual_scene",
    "_extract_chart_text_structure",
    "_format_chart_text_details",
    "_format_layout_details",
    "_looks_like_url_or_query",
    "_match_common_ui_text",
    "_merge_ocr_lines",
    "_normalize_ocr_lines",
    "_normalize_short_text_list",
    "_ocr_line_text_quality",
    "_repair_short_ui_text",
    "_repair_url_like_text",
    "_stabilize_layout_text",
    "_structured_summary_labels",
]
