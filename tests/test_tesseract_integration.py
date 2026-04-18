import base64
import io
import os
import shutil
import subprocess
import sys
import unittest
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import vision_local as vision_local_module


FIXTURES_DIR = ROOT / "tests" / "fixtures"


def _fixture_path(name: str) -> Path:
    return FIXTURES_DIR / name


def _load_fixture_image(name: str) -> Image.Image:
    with Image.open(_fixture_path(name)) as image:
        return image.convert("RGB")


def _fixture_b64(name: str) -> str:
    with _fixture_path(name).open("rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


@lru_cache(maxsize=1)
def _tesseract_languages() -> set[str]:
    binary = shutil.which("tesseract")
    if not binary:
        return set()
    try:
        completed = subprocess.run(
            [binary, "--list-langs"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return set()
    output = "\n".join(filter(None, [completed.stdout, completed.stderr]))
    return {line.strip() for line in output.splitlines() if line.strip() and "languages" not in line.lower()}


def _preferred_tesseract_lang() -> str:
    languages = _tesseract_languages()
    if "chi_sim" in languages and "eng" in languages:
        return "eng+chi_sim"
    if "eng" in languages:
        return "eng"
    return ""


def _lowered_text(values: list[dict]) -> str:
    return " ".join(str(item.get("text", "")) for item in values).lower()


class TesseractFixtureSmokeTests(unittest.TestCase):
    def test_checked_in_png_fixtures_load_with_expected_sizes(self):
        expected = {
            "mixed_browser_login.png": (1366, 820),
            "mixed_chat_low_res.png": (640, 360),
        }
        for name, size in expected.items():
            path = _fixture_path(name)
            self.assertTrue(path.exists(), path)
            with Image.open(path) as image:
                self.assertEqual((image.width, image.height), size)


@unittest.skipUnless(vision_local_module.has_tesseract_ocr_support(), "tesseract CLI not available")
class TesseractIntegrationTests(unittest.TestCase):
    def test_tesseract_backend_extracts_browser_fixture_text_and_lines(self):
        image = _load_fixture_image("mixed_browser_login.png")

        with patch.dict(
            os.environ,
            {"VISION_LOCAL_CONTEXT_TESSERACT_LANG": _preferred_tesseract_lang()},
            clear=False,
        ):
            result = vision_local_module._run_tesseract_ocr(
                image,
                lambda *_args: None,
                include_layout=True,
            )

        lowered = str(result["text"]).lower()
        self.assertGreaterEqual(len(result["lines"]), 4)
        self.assertTrue(any(token in lowered for token in ("example", "login", "email", "password")))
        self.assertTrue(any(token in _lowered_text(result["lines"]) for token in ("example", "login", "password")))

    def test_analyze_image_reports_tesseract_backend_on_real_browser_fixture(self):
        image_b64 = _fixture_b64("mixed_browser_login.png")

        with patch.dict(
            os.environ,
            {
                "VISION_LOCAL_CONTEXT_OCR_BACKEND": "tesseract",
                "VISION_LOCAL_CONTEXT_TESSERACT_LANG": _preferred_tesseract_lang(),
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
        self.assertEqual(result["scene"], "browser")
        self.assertEqual(result["layout"]["kind"], "browser")
        self.assertIn("example.cn", result["layout"]["address_bar"])
        self.assertTrue(
            any(
                token in result["visible_text"].lower()
                for token in ("login", "email", "password", "example.cn")
            )
        )

    def test_analyze_image_detects_chat_layout_on_real_low_res_fixture(self):
        image_b64 = _fixture_b64("mixed_chat_low_res.png")

        with patch.dict(
            os.environ,
            {
                "VISION_LOCAL_CONTEXT_OCR_BACKEND": "tesseract",
                "VISION_LOCAL_CONTEXT_TESSERACT_LANG": _preferred_tesseract_lang(),
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
        self.assertEqual(result["scene"], "chat")
        self.assertEqual(result["layout"]["kind"], "chat")
        self.assertTrue(str(result["layout"].get("title") or "").strip())
        self.assertTrue(str(result["layout"].get("input_hint") or "").strip())
        self.assertIn("chat or messaging interface", result["summary"])
        self.assertTrue(
            any(
                token in result["visible_text"].lower()
                for token in ("reply", "design", "product", "chat")
            )
        )

    @unittest.skipUnless("chi_sim" in _tesseract_languages(), "tesseract chi_sim language pack not available")
    def test_mixed_language_browser_fixture_preserves_cjk_signal(self):
        image = _load_fixture_image("mixed_browser_login.png")

        with patch.dict(
            os.environ,
            {"VISION_LOCAL_CONTEXT_TESSERACT_LANG": "eng+chi_sim"},
            clear=False,
        ):
            result = vision_local_module._run_tesseract_ocr(
                image,
                lambda *_args: None,
                include_layout=False,
            )

        text = str(result)
        self.assertTrue(any(token in text for token in ("\u793a\u4f8b", "\u90ae\u7bb1", "\u5bc6\u7801", "\u767b\u5f55")))


if __name__ == "__main__":
    unittest.main()
