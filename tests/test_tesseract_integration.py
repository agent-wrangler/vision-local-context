import base64
import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import vision_local as vision_local_module


def _load_test_font(size: int):
    candidates = [
        "DejaVuSans.ttf",
        "Arial.ttf",
        "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _make_tesseract_browser_fixture() -> Image.Image:
    image = Image.new("RGB", (1400, 820), "white")
    draw = ImageDraw.Draw(image)
    title_font = _load_test_font(58)
    body_font = _load_test_font(42)
    button_font = _load_test_font(44)

    draw.rounded_rectangle((60, 40, 1340, 780), radius=24, fill="#FFFFFF", outline="#CBD5E1", width=3)
    draw.rounded_rectangle((90, 64, 1240, 116), radius=14, fill="#EEF2F7", outline="#CBD5E1", width=2)
    draw.text((124, 74), "https://example.com/login", fill="#111827", font=body_font)
    draw.text((492, 186), "Example Login", fill="#111827", font=title_font)
    draw.text((422, 306), "Email", fill="#111827", font=body_font)
    draw.rounded_rectangle((408, 346, 992, 412), radius=10, outline="#94A3B8", width=2)
    draw.text((422, 446), "Password", fill="#111827", font=body_font)
    draw.rounded_rectangle((408, 486, 992, 552), radius=10, outline="#94A3B8", width=2)
    draw.rounded_rectangle((520, 608, 878, 680), radius=12, fill="#2563EB")
    draw.text((618, 624), "Sign in", fill="#FFFFFF", font=button_font)
    return image


@unittest.skipUnless(vision_local_module.has_tesseract_ocr_support(), "tesseract CLI not available")
class TesseractIntegrationTests(unittest.TestCase):
    def test_tesseract_backend_extracts_real_text_and_lines(self):
        image = _make_tesseract_browser_fixture()

        result = vision_local_module._run_tesseract_ocr(
            image,
            lambda *_args: None,
            include_layout=True,
        )

        self.assertIn("Example", result["text"])
        self.assertGreaterEqual(len(result["lines"]), 3)
        joined = " ".join(line["text"] for line in result["lines"])
        self.assertIn("Password", joined)

    def test_analyze_image_reports_tesseract_backend_on_real_fixture(self):
        image = _make_tesseract_browser_fixture()
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

        with patch.dict(
            os.environ,
            {
                "VISION_LOCAL_CONTEXT_OCR_BACKEND": "tesseract",
                "VISION_LOCAL_CONTEXT_TESSERACT_LANG": "eng",
            },
            clear=False,
        ), patch.object(
            vision_local_module,
            "_should_attempt_caption",
            return_value=False,
        ):
            result = vision_local_module.analyze_image(image_b64, debug_write=lambda *_args: None)

        self.assertTrue(result["ok"])
        self.assertEqual(result["ocr_backend"], "tesseract")
        self.assertIn("Example", result["ocr_text"])


if __name__ == "__main__":
    unittest.main()
