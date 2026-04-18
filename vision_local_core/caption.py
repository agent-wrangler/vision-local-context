from __future__ import annotations

import time

from PIL import Image

from . import _legacy, summary as _summary

_DEBUG_WRITE = _legacy._DEBUG_WRITE
_env_flag = _legacy._env_flag
_normalize_text = _legacy._normalize_text
contextlib = _legacy.contextlib
io = _legacy.io
os = _legacy.os
threading = _legacy.threading


def _caption_blocking_enabled() -> bool:
    return _env_flag("VISION_LOCAL_CONTEXT_CAPTION_BLOCKING", default=False)


def _caption_allow_download() -> bool:
    return _env_flag("VISION_LOCAL_CONTEXT_CAPTION_ALLOW_DOWNLOAD", default=False)


def _get_caption_backend_state() -> dict[str, bool]:
    with _legacy._CAPTION_LOCK:
        return {
            "ready": _legacy._CAPTION_BACKEND is not None,
            "loading": _legacy._CAPTION_LOADING,
            "error": bool(_legacy._CAPTION_LOAD_ERROR),
        }


def _run_caption_analysis(image: Image.Image, ocr_text: str, debug_write: _DEBUG_WRITE) -> dict:
    if not _summary._should_attempt_caption(image, ocr_text):
        return {
            "caption": "",
            "pending": False,
            "ms": 0.0,
        }

    caption_started_at = time.perf_counter()
    caption = _caption_image(
        image,
        debug_write,
        prompt=_summary._choose_caption_prompt(image, ocr_text),
    )
    caption_ms = round((time.perf_counter() - caption_started_at) * 1000, 1)
    state = _get_caption_backend_state()
    return {
        "caption": caption,
        "pending": not caption and state["loading"] and not state["error"],
        "ms": caption_ms,
    }


def _ensure_caption_backend_loading(debug_write: _DEBUG_WRITE) -> None:
    if _legacy._CAPTION_BACKEND is not None:
        return
    if _legacy._CAPTION_LOAD_ATTEMPTED and _legacy._CAPTION_LOAD_ERROR:
        return
    with _legacy._CAPTION_LOCK:
        if _legacy._CAPTION_BACKEND is not None:
            return
        if _legacy._CAPTION_LOAD_ATTEMPTED and _legacy._CAPTION_LOAD_ERROR:
            return
        if _legacy._CAPTION_LOADING:
            return
        _legacy._CAPTION_LOADING = True

    def _runner() -> None:
        try:
            _load_caption_backend(debug_write, blocking=True)
        finally:
            with _legacy._CAPTION_LOCK:
                _legacy._CAPTION_LOADING = False

    thread = threading.Thread(target=_runner, name="vision-caption-loader", daemon=True)
    thread.start()
    debug_write("vision_local_caption_loading", {"background": True})


def _load_caption_backend(debug_write: _DEBUG_WRITE, *, blocking: bool = False) -> tuple[object, object] | None:
    if _legacy._CAPTION_BACKEND is not None:
        return _legacy._CAPTION_BACKEND
    if _legacy._CAPTION_LOAD_ATTEMPTED and _legacy._CAPTION_LOAD_ERROR:
        return None
    if not blocking:
        _ensure_caption_backend_loading(debug_write)
        return _legacy._CAPTION_BACKEND

    with _legacy._CAPTION_LOCK:
        if _legacy._CAPTION_BACKEND is not None:
            return _legacy._CAPTION_BACKEND
        if _legacy._CAPTION_LOAD_ATTEMPTED and _legacy._CAPTION_LOAD_ERROR:
            return None
        _legacy._CAPTION_LOAD_ATTEMPTED = True

    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor
        from transformers.utils import logging as transformers_logging

        model_id = (
            os.environ.get("VISION_LOCAL_CONTEXT_CAPTION_MODEL")
            or "Salesforce/blip-image-captioning-base"
        ).strip()
        local_only = not _caption_allow_download()
        transformers_logging.set_verbosity_error()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            processor = BlipProcessor.from_pretrained(model_id, local_files_only=local_only)
            model = BlipForConditionalGeneration.from_pretrained(model_id, local_files_only=local_only)
        model.eval()
        with _legacy._CAPTION_LOCK:
            _legacy._CAPTION_BACKEND = (processor, model)
            _legacy._CAPTION_LOAD_ERROR = ""
        debug_write(
            "vision_local_caption_ready",
            {"model_id": model_id, "local_only": local_only},
        )
    except Exception as exc:
        with _legacy._CAPTION_LOCK:
            _legacy._CAPTION_BACKEND = None
            _legacy._CAPTION_LOAD_ERROR = str(exc)
        debug_write("vision_local_caption_unavailable", {"error": str(exc)[:240]})
    return _legacy._CAPTION_BACKEND


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


def has_caption_support() -> bool:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:
        return False


__all__ = [
    "_caption_allow_download",
    "_caption_blocking_enabled",
    "_caption_image",
    "_ensure_caption_backend_loading",
    "_get_caption_backend_state",
    "_load_caption_backend",
    "_run_caption_analysis",
    "has_caption_support",
]
