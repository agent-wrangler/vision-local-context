from __future__ import annotations

from ._legacy import (
    _caption_allow_download,
    _caption_blocking_enabled,
    _caption_image,
    _ensure_caption_backend_loading,
    _get_caption_backend_state,
    _load_caption_backend,
    has_caption_support,
)

__all__ = [
    "_caption_allow_download",
    "_caption_blocking_enabled",
    "_caption_image",
    "_ensure_caption_backend_loading",
    "_get_caption_backend_state",
    "_load_caption_backend",
    "has_caption_support",
]
