from __future__ import annotations

import base64
import hashlib
import io
import time

from PIL import Image

from . import _legacy
from .caption import _load_caption_backend, _run_caption_analysis
from .layout import (
    _analyze_chart_visual_pattern,
    _analyze_structured_layout,
    _build_clean_visible_text,
    _detect_visual_scene,
    _extract_chart_text_structure,
    _format_chart_text_details,
    _format_layout_details,
)
from .ocr import _run_ocr_analysis
from .summary import _build_visual_summary

_DEBUG_WRITE = _legacy._DEBUG_WRITE
_normalize_text = _legacy._normalize_text


def _noop_debug(_stage: str, _data: dict) -> None:
    return None


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
    with _legacy._CACHE_LOCK:
        cached = _legacy._ANALYSIS_CACHE.get(digest)
        if cached is None:
            return None
        _legacy._ANALYSIS_CACHE.move_to_end(digest)
        return dict(cached)


def _cache_put(digest: str, analysis: dict) -> None:
    if not digest:
        return
    with _legacy._CACHE_LOCK:
        _legacy._ANALYSIS_CACHE[digest] = dict(analysis)
        _legacy._ANALYSIS_CACHE.move_to_end(digest)
        while len(_legacy._ANALYSIS_CACHE) > _legacy._CACHE_LIMIT:
            _legacy._ANALYSIS_CACHE.popitem(last=False)


def _empty_analysis(
    *,
    summary: str,
    ok: bool = False,
    digest: str = "",
    size: str = "",
) -> dict:
    return {
        "ok": ok,
        "caption": "",
        "ocr_text": "",
        "visible_text": "",
        "summary": summary,
        "scene": "",
        "chart_visual": {},
        "chart_text": {},
        "layout": {},
        "digest": digest,
        "size": size,
        "ocr_backend": "",
        "caption_pending": False,
    }


def analyze_image(image_b64: str, *, debug_write: _DEBUG_WRITE | None = None) -> dict:
    debug_write = debug_write or _noop_debug
    started_at = time.perf_counter()
    image, raw = _decode_image(image_b64)
    if image is None or not raw:
        return _empty_analysis(summary="Unable to decode the uploaded image.")

    digest = hashlib.sha256(raw).hexdigest()
    cached = _cache_get(digest)
    if cached is not None:
        return cached

    if image.width >= 900 and image.height >= 500:
        _load_caption_backend(debug_write, blocking=False)

    ocr_result = _run_ocr_analysis(image, debug_write)
    ocr_text = ocr_result["text"]
    ocr_lines = ocr_result["lines"]
    chart_visual = _analyze_chart_visual_pattern(image)
    pre_scene = _detect_visual_scene(image, caption="", ocr_text=ocr_text, chart_visual=chart_visual)
    layout = _analyze_structured_layout(image, ocr_lines, ocr_text)
    chart_text = _extract_chart_text_structure(image, ocr_lines) if ocr_lines and pre_scene == "chart" else {}

    caption_result = _run_caption_analysis(image, ocr_text, debug_write)
    caption = caption_result["caption"]
    caption_pending = caption_result["pending"]

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
        "ocr_backend": ocr_result["backend"],
        "caption_pending": caption_pending,
    }
    if not caption_pending:
        _cache_put(digest, analysis)
    debug_write(
        "vision_local_analysis_timing",
        {
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 1),
            "ocr_ms": ocr_result["ms"],
            "caption_ms": caption_result["ms"],
            "has_ocr": bool(ocr_text),
            "has_caption": bool(caption),
            "ocr_backend": ocr_result["backend"],
            "ocr_retried": ocr_result["retried"],
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


__all__ = [
    "_cache_get",
    "_cache_put",
    "_decode_image",
    "_empty_analysis",
    "_noop_debug",
    "_normalize_text",
    "_run_caption_analysis",
    "_run_ocr_analysis",
    "analyze_image",
    "build_screen_description",
    "build_user_image_context",
]
