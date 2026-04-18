from __future__ import annotations

import csv
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import OrderedDict

from PIL import Image, ImageFilter, ImageOps

from . import caption as _caption
from .contracts import LocalImageCapabilities, OCRAnalysisResult
from . import layout as _layout
from .shared import (
    _DEBUG_WRITE,
    _RESAMPLING,
    _env_int,
    _extract_readable_labels,
    _is_low_signal_ocr,
    _normalize_text,
    _ocr_line_text_quality,
    _ocr_signal_score,
)


def _ocr_timeout_seconds() -> int:
    return _env_int("VISION_LOCAL_CONTEXT_OCR_TIMEOUT_SECONDS", 8, minimum=3, maximum=20)


def _ocr_backend_preference() -> str:
    raw = str(os.environ.get("VISION_LOCAL_CONTEXT_OCR_BACKEND", "") or "").strip().lower()
    if raw in {"windows", "tesseract"}:
        return raw
    return "auto"


def _tesseract_psm() -> int:
    return _env_int("VISION_LOCAL_CONTEXT_TESSERACT_PSM", 11, minimum=3, maximum=13)


def _tesseract_lang() -> str:
    return _normalize_text(os.environ.get("VISION_LOCAL_CONTEXT_TESSERACT_LANG", ""), limit=48).replace(" ", "")


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


def _empty_ocr_result(*, include_layout: bool) -> str | dict:
    if include_layout:
        return {"text": "", "lines": []}
    return ""


def _run_local_ocr_with_backend(
    image: Image.Image,
    debug_write: _DEBUG_WRITE,
    *,
    include_layout: bool = False,
) -> tuple[str | dict, str]:
    preference = _ocr_backend_preference()
    backends: list[str]
    if preference == "windows":
        backends = ["windows"]
    elif preference == "tesseract":
        backends = ["tesseract"]
    else:
        backends = ["windows", "tesseract"]

    for backend in backends:
        if backend == "windows" and has_windows_ocr_support():
            return _run_windows_ocr(image, debug_write, include_layout=include_layout), "windows"
        if backend == "tesseract" and has_tesseract_ocr_support():
            return _run_tesseract_ocr(image, debug_write, include_layout=include_layout), "tesseract"
    return _empty_ocr_result(include_layout=include_layout), ""


def _run_ocr_analysis(image: Image.Image, debug_write: _DEBUG_WRITE) -> OCRAnalysisResult:
    ocr_started_at = time.perf_counter()
    ocr_payload, ocr_backend = _run_local_ocr_with_backend(image, debug_write, include_layout=True)
    if isinstance(ocr_payload, dict):
        ocr_text = _normalize_text(ocr_payload.get("text", ""), limit=1600)
        ocr_lines = _layout._normalize_ocr_lines(image, ocr_payload.get("lines", []))
    else:
        ocr_text = _normalize_text(ocr_payload, limit=1600)
        ocr_lines = []

    ocr_retried = False
    if _should_retry_ocr(image, ocr_text):
        retry_image = _build_ocr_retry_image(image)
        retry_payload, retry_backend = _run_local_ocr_with_backend(retry_image, debug_write, include_layout=True)
        ocr_retried = True
        if isinstance(retry_payload, dict):
            retry_text = _normalize_text(retry_payload.get("text", ""), limit=1600)
            retry_lines = _layout._normalize_ocr_lines(
                image,
                retry_payload.get("lines", []),
                source_size=(retry_image.width, retry_image.height),
            )
        else:
            retry_text = _normalize_text(retry_payload, limit=1600)
            retry_lines = []
        if retry_backend:
            ocr_backend = retry_backend
        if retry_lines:
            ocr_lines = _layout._merge_ocr_lines(ocr_lines, retry_lines)
        if _ocr_signal_score(retry_text) > _ocr_signal_score(ocr_text):
            ocr_text = retry_text

    return {
        "text": ocr_text,
        "lines": ocr_lines,
        "retried": ocr_retried,
        "backend": ocr_backend,
        "ms": round((time.perf_counter() - ocr_started_at) * 1000, 1),
    }
def _parse_tesseract_tsv(payload: str) -> dict:
    reader = csv.DictReader(io.StringIO(str(payload or "")), delimiter="\t")
    line_map: OrderedDict[tuple[str, str, str, str], dict] = OrderedDict()

    def _safe_int(value) -> int:
        try:
            return int(float(value or 0))
        except Exception:
            return 0

    for row in reader:
        level = _safe_int(row.get("level"))
        if level not in {4, 5}:
            continue
        key = (
            str(row.get("page_num") or ""),
            str(row.get("block_num") or ""),
            str(row.get("par_num") or ""),
            str(row.get("line_num") or ""),
        )
        left = _safe_int(row.get("left"))
        top = _safe_int(row.get("top"))
        width = _safe_int(row.get("width"))
        height = _safe_int(row.get("height"))
        right = left + max(width, 0)
        bottom = top + max(height, 0)
        entry = line_map.setdefault(
            key,
            {
                "words": [],
                "x": left,
                "y": top,
                "right": right,
                "bottom": bottom,
            },
        )
        if level == 4:
            entry["x"] = left
            entry["y"] = top
            entry["right"] = max(entry["right"], right)
            entry["bottom"] = max(entry["bottom"], bottom)
            continue

        word = _normalize_text(row.get("text", ""), limit=80)
        if word:
            entry["words"].append(word)
        if entry["words"]:
            entry["x"] = min(entry["x"], left)
            entry["y"] = min(entry["y"], top)
            entry["right"] = max(entry["right"], right)
            entry["bottom"] = max(entry["bottom"], bottom)

    lines: list[dict] = []
    for entry in line_map.values():
        text = _normalize_text(" ".join(entry["words"]), limit=220)
        if not text:
            continue
        x = max(int(entry["x"]), 0)
        y = max(int(entry["y"]), 0)
        right = max(int(entry["right"]), x + 1)
        bottom = max(int(entry["bottom"]), y + 1)
        lines.append(
            {
                "text": text,
                "x": x,
                "y": y,
                "width": right - x,
                "height": bottom - y,
            }
        )
    lines.sort(key=lambda item: (item["y"], item["x"]))
    return {
        "text": _normalize_text(" ".join(line["text"] for line in lines), limit=1600),
        "lines": lines,
    }


def _run_tesseract_ocr(image: Image.Image, debug_write: _DEBUG_WRITE, *, include_layout: bool = False) -> str | dict:
    tesseract = shutil.which("tesseract")
    if not tesseract:
        return _empty_ocr_result(include_layout=include_layout)
    tmp_path = ""
    timeout_seconds = _ocr_timeout_seconds()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as handle:
            tmp_path = handle.name
        image.save(tmp_path, format="PNG", optimize=True)
        args = [tesseract, tmp_path, "-", "--psm", str(_tesseract_psm())]
        lang = _tesseract_lang()
        if lang:
            args.extend(["-l", lang])
        if include_layout:
            args.extend(["tsv", "quiet"])
        else:
            args.append("quiet")
        completed = subprocess.run(
            args,
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
                debug_write("vision_local_tesseract_error", {"error": stderr})
            return _empty_ocr_result(include_layout=include_layout)
        stdout = str(completed.stdout or "")
        if include_layout:
            return _parse_tesseract_tsv(stdout)
        return _normalize_text(stdout, limit=1600)
    except subprocess.TimeoutExpired:
        debug_write("vision_local_tesseract_timeout", {"timeout_s": timeout_seconds})
        return _empty_ocr_result(include_layout=include_layout)
    except Exception as exc:
        debug_write("vision_local_tesseract_exception", {"error": str(exc)[:240]})
        return _empty_ocr_result(include_layout=include_layout)
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _run_local_ocr(image: Image.Image, debug_write: _DEBUG_WRITE, *, include_layout: bool = False) -> str | dict:
    payload, _backend = _run_local_ocr_with_backend(image, debug_write, include_layout=include_layout)
    return payload


def _run_windows_ocr(image: Image.Image, debug_write: _DEBUG_WRITE, *, include_layout: bool = False) -> str | dict:
    if os.name != "nt":
        return _empty_ocr_result(include_layout=include_layout)
    powershell = shutil.which("powershell")
    if not powershell:
        return _empty_ocr_result(include_layout=include_layout)
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
            return _empty_ocr_result(include_layout=include_layout)
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
        return _empty_ocr_result(include_layout=include_layout)
    except Exception as exc:
        debug_write("vision_local_ocr_exception", {"error": str(exc)})
        return _empty_ocr_result(include_layout=include_layout)
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def has_windows_ocr_support() -> bool:
    return os.name == "nt" and bool(shutil.which("powershell"))


def has_tesseract_ocr_support() -> bool:
    return bool(shutil.which("tesseract"))


def get_local_image_capabilities() -> LocalImageCapabilities:
    windows_ocr = has_windows_ocr_support()
    tesseract_ocr = has_tesseract_ocr_support()
    caption = _caption.has_caption_support()
    return {
        "windows_ocr": windows_ocr,
        "tesseract_ocr": tesseract_ocr,
        "caption": caption,
        "full_analysis": windows_ocr or tesseract_ocr,
        "any": windows_ocr or tesseract_ocr or caption,
    }


def has_local_image_support() -> bool:
    return has_windows_ocr_support() or has_tesseract_ocr_support()


__all__ = [
    "_build_ocr_retry_image",
    "_empty_ocr_result",
    "_is_low_signal_ocr",
    "_ocr_backend_preference",
    "_ocr_line_text_quality",
    "_ocr_signal_score",
    "_ocr_timeout_seconds",
    "_parse_tesseract_tsv",
    "_run_local_ocr",
    "_run_local_ocr_with_backend",
    "_run_ocr_analysis",
    "_run_tesseract_ocr",
    "_run_windows_ocr",
    "_should_retry_ocr",
    "_tesseract_lang",
    "_tesseract_psm",
    "get_local_image_capabilities",
    "has_local_image_support",
    "has_tesseract_ocr_support",
    "has_windows_ocr_support",
]
