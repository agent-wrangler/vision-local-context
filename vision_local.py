from __future__ import annotations

from vision_local_core import (
    _legacy,
    caption as _caption,
    layout as _layout,
    ocr as _ocr,
    pipeline as _pipeline,
    summary as _summary,
)
from vision_local_core.caption import *
from vision_local_core.layout import *
from vision_local_core.ocr import *
from vision_local_core.pipeline import *
from vision_local_core.summary import *
from vision_local_core._legacy import (
    __all__ as _PUBLIC_API,
    _CAPTION_BACKEND,
    _CAPTION_LOAD_ATTEMPTED,
    _CAPTION_LOAD_ERROR,
    _CAPTION_LOADING,
    _DEBUG_WRITE,
    os,
    shutil,
    subprocess,
    tempfile,
)

__all__ = list(_PUBLIC_API)

_SYNC_EXCLUDED = {
    "__all__",
    "_PUBLIC_API",
    "_SYNC_EXCLUDED",
    "_SYNC_TARGETS",
    "_WRAPPER_PASSTHROUGH",
    "_WRAPPER_ORIGINALS",
    "_caption",
    "_layout",
    "_legacy",
    "_ocr",
    "_pipeline",
    "_summary",
    "_sync_legacy_globals",
}

_SYNC_TARGETS = (_legacy, _caption, _layout, _ocr, _pipeline, _summary)


def _sync_legacy_globals(*, include_analyze: bool = False) -> None:
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
    _sync_legacy_globals()
    return _legacy.analyze_image(image_b64, debug_write=debug_write)


def _run_local_ocr_with_backend(
    image,
    debug_write: _DEBUG_WRITE,
    *,
    include_layout: bool = False,
):
    _sync_legacy_globals()
    return _ocr._run_local_ocr_with_backend(image, debug_write, include_layout=include_layout)


def _run_local_ocr(
    image,
    debug_write: _DEBUG_WRITE,
    *,
    include_layout: bool = False,
):
    _sync_legacy_globals()
    return _ocr._run_local_ocr(image, debug_write, include_layout=include_layout)


def _caption_image(image, debug_write: _DEBUG_WRITE, *, prompt: str = "") -> str:
    _sync_legacy_globals()
    return _caption._caption_image(image, debug_write, prompt=prompt)


def build_user_image_context(
    images: list[str] | None,
    *,
    user_text: str = "",
    debug_write: _DEBUG_WRITE | None = None,
) -> str:
    _sync_legacy_globals(include_analyze=True)
    return _legacy.build_user_image_context(images, user_text=user_text, debug_write=debug_write)


def build_screen_description(image_b64: str, *, debug_write: _DEBUG_WRITE | None = None) -> str:
    _sync_legacy_globals(include_analyze=True)
    return _legacy.build_screen_description(image_b64, debug_write=debug_write)


def get_local_image_capabilities() -> dict[str, bool]:
    _sync_legacy_globals()
    return _ocr.get_local_image_capabilities()


def has_local_image_support() -> bool:
    _sync_legacy_globals()
    return _ocr.has_local_image_support()


_WRAPPER_PASSTHROUGH = {
    "analyze_image": analyze_image,
    "_run_local_ocr_with_backend": _run_local_ocr_with_backend,
    "_run_local_ocr": _run_local_ocr,
    "_caption_image": _caption_image,
    "build_user_image_context": build_user_image_context,
    "build_screen_description": build_screen_description,
    "get_local_image_capabilities": get_local_image_capabilities,
    "has_local_image_support": has_local_image_support,
}

_WRAPPER_ORIGINALS = {
    "analyze_image": _legacy.analyze_image,
    "_run_local_ocr_with_backend": _ocr._run_local_ocr_with_backend,
    "_run_local_ocr": _ocr._run_local_ocr,
    "_caption_image": _caption._caption_image,
    "build_user_image_context": _legacy.build_user_image_context,
    "build_screen_description": _legacy.build_screen_description,
    "get_local_image_capabilities": _ocr.get_local_image_capabilities,
    "has_local_image_support": _ocr.has_local_image_support,
}
