from __future__ import annotations

from typing import Literal, TypedDict


class ChartBoundingBox(TypedDict):
    width: int
    height: int


class ChartVisualResult(TypedDict, total=False):
    chart_like: bool
    kind: Literal["bar", "line"]
    trend: Literal["upward", "downward", ""]
    series_count: int
    bbox: ChartBoundingBox


class ChartTextResult(TypedDict, total=False):
    title: str
    x_axis_labels: list[str]
    y_axis_labels: list[str]
    legend_labels: list[str]


class LayoutResult(TypedDict, total=False):
    kind: Literal["browser", "chat"]
    address_bar: str
    page_title: str
    field_labels: list[str]
    title: str
    left_messages: list[str]
    right_messages: list[str]
    input_hint: str
    sidebar_labels: list[str]


class ImageAnalysisResult(TypedDict):
    ok: bool
    caption: str
    ocr_text: str
    visible_text: str
    summary: str
    scene: str
    chart_visual: ChartVisualResult
    chart_text: ChartTextResult
    layout: LayoutResult
    digest: str
    size: str
    ocr_backend: str
    caption_pending: bool


class LocalImageCapabilities(TypedDict):
    windows_ocr: bool
    tesseract_ocr: bool
    caption: bool
    full_analysis: bool
    any: bool


class OCRAnalysisResult(TypedDict):
    text: str
    lines: list[dict]
    retried: bool
    backend: str
    ms: float


class CaptionAnalysisResult(TypedDict):
    caption: str
    pending: bool
    ms: float


__all__ = [
    "CaptionAnalysisResult",
    "ChartBoundingBox",
    "ChartTextResult",
    "ChartVisualResult",
    "ImageAnalysisResult",
    "LayoutResult",
    "LocalImageCapabilities",
    "OCRAnalysisResult",
]
