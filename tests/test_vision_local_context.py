import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import vision_local as vision_local_module


class _FakeImage:
    width = 1280
    height = 720

    def convert(self, *_args, **_kwargs):
        return self

    def resize(self, *_args, **_kwargs):
        return self

    def filter(self, *_args, **_kwargs):
        return self

    def save(self, *_args, **_kwargs):
        return None


class _FakeTempFile:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _make_line_chart_image():
    image = Image.new("RGB", (1280, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((40, 40, 1240, 760), radius=18, fill="#ffffff", outline="#d1d5db", width=2)
    draw.line((120, 640, 1160, 640), fill="#9ca3af", width=3)
    draw.line((120, 180, 120, 640), fill="#9ca3af", width=3)
    points = [(260, 560), (470, 430), (680, 300), (890, 220)]
    draw.line(points, fill="#2563eb", width=6)
    for x, y in points:
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill="#2563eb")
    return image


def _make_bar_chart_image():
    image = Image.new("RGB", (1280, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((40, 40, 1240, 760), radius=18, fill="#ffffff", outline="#d1d5db", width=2)
    draw.line((120, 640, 1160, 640), fill="#9ca3af", width=3)
    draw.line((120, 180, 120, 640), fill="#9ca3af", width=3)
    for x, top in [(220, 500), (450, 400), (680, 300), (910, 220)]:
        draw.rounded_rectangle((x, top, x + 90, 640), radius=8, fill="#10b981")
    return image


def _make_settings_like_image():
    image = Image.new("RGB", (1280, 800), "#f3f4f6")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((40, 40, 1240, 760), radius=18, fill="white", outline="#d1d5db", width=2)
    draw.rectangle((40, 40, 1240, 110), fill="#111827")
    draw.rounded_rectangle((720, 214, 930, 254), radius=20, fill="#dbeafe")
    draw.rounded_rectangle((720, 300, 820, 348), radius=24, fill="#22c55e")
    draw.rectangle((720, 410, 1040, 418), fill="#d1d5db")
    draw.rectangle((720, 410, 920, 418), fill="#3b82f6")
    draw.ellipse((910, 398, 934, 422), fill="#2563eb")
    return image


def _browser_layout_lines():
    return [
        {"text": "https://example.com/login", "x": 160, "y": 26, "width": 540, "height": 24},
        {"text": "Example Login", "x": 220, "y": 118, "width": 220, "height": 24},
        {"text": "Email", "x": 230, "y": 250, "width": 80, "height": 20},
        {"text": "Password", "x": 230, "y": 320, "width": 110, "height": 20},
        {"text": "Sign in", "x": 230, "y": 390, "width": 90, "height": 20},
    ]


def _chat_layout_lines():
    return [
        {"text": "Alice", "x": 560, "y": 38, "width": 80, "height": 24},
        {"text": "General", "x": 40, "y": 120, "width": 80, "height": 18},
        {"text": "Design", "x": 40, "y": 190, "width": 70, "height": 18},
        {"text": "Can you review this?", "x": 120, "y": 210, "width": 220, "height": 20},
        {"text": "Sure, send it over", "x": 760, "y": 280, "width": 220, "height": 20},
        {"text": "Type a message", "x": 420, "y": 660, "width": 180, "height": 20},
    ]


def _chart_axis_lines():
    return [
        {"text": "Revenue Dashboard", "x": 80, "y": 70, "width": 240, "height": 24},
        {"text": "80k", "x": 60, "y": 250, "width": 40, "height": 18},
        {"text": "60k", "x": 60, "y": 360, "width": 40, "height": 18},
        {"text": "40k", "x": 60, "y": 470, "width": 40, "height": 18},
        {"text": "20k", "x": 60, "y": 580, "width": 40, "height": 18},
        {"text": "Q1", "x": 260, "y": 660, "width": 30, "height": 18},
        {"text": "Q2", "x": 470, "y": 660, "width": 30, "height": 18},
        {"text": "Q3", "x": 680, "y": 660, "width": 30, "height": 18},
        {"text": "Q4", "x": 890, "y": 660, "width": 30, "height": 18},
        {"text": "Growth +18%", "x": 980, "y": 135, "width": 120, "height": 18},
    ]


class VisionLocalBridgeTests(unittest.TestCase):
    def test_public_api_exports_supported_entry_points(self):
        self.assertEqual(
            vision_local_module.__all__,
            [
                "analyze_image",
                "build_screen_description",
                "build_user_image_context",
                "get_local_image_capabilities",
                "has_caption_support",
                "has_local_image_support",
                "has_tesseract_ocr_support",
                "has_windows_ocr_support",
            ],
        )

    def test_analyze_image_decode_failure_returns_stable_shape(self):
        with patch.object(vision_local_module, "_decode_image", return_value=(None, b"")):
            result = vision_local_module.analyze_image("not-base64")

        self.assertEqual(
            result,
            {
                "ok": False,
                "caption": "",
                "ocr_text": "",
                "visible_text": "",
                "summary": "Unable to decode the uploaded image.",
                "scene": "",
                "chart_visual": {},
                "chart_text": {},
                "layout": {},
                "digest": "",
                "size": "",
                "caption_pending": False,
            },
        )

    def test_has_windows_ocr_support_requires_windows_and_powershell(self):
        with patch.object(vision_local_module.os, "name", "nt"), patch.object(
            vision_local_module.shutil, "which", return_value="powershell"
        ):
            self.assertTrue(vision_local_module.has_windows_ocr_support())

        with patch.object(vision_local_module.os, "name", "posix"), patch.object(
            vision_local_module.shutil, "which", return_value=None
        ):
            self.assertFalse(vision_local_module.has_windows_ocr_support())

    def test_has_caption_support_checks_optional_caption_dependencies(self):
        with patch.dict(sys.modules, {"transformers": object(), "torch": object()}):
            self.assertTrue(vision_local_module.has_caption_support())

    def test_has_tesseract_ocr_support_checks_binary_presence(self):
        with patch.object(vision_local_module.shutil, "which", return_value="tesseract"):
            self.assertTrue(vision_local_module.has_tesseract_ocr_support())
        with patch.object(vision_local_module.shutil, "which", return_value=None):
            self.assertFalse(vision_local_module.has_tesseract_ocr_support())

    def test_has_local_image_support_accepts_tesseract_fallback(self):
        with patch.object(vision_local_module, "has_windows_ocr_support", return_value=False), patch.object(
            vision_local_module, "has_tesseract_ocr_support", return_value=True
        ), patch.object(
            vision_local_module, "has_caption_support", return_value=True
        ):
            self.assertTrue(vision_local_module.has_local_image_support())
            self.assertEqual(
                vision_local_module.get_local_image_capabilities(),
                {
                    "windows_ocr": False,
                    "tesseract_ocr": True,
                    "caption": True,
                    "full_analysis": True,
                    "any": True,
                },
            )

    def test_standalone_environment_variable_prefix_is_used(self):
        with patch.dict(
            vision_local_module.os.environ,
            {
                "VISION_LOCAL_CONTEXT_OCR_TIMEOUT_SECONDS": "11",
                "VISION_LOCAL_CONTEXT_OCR_BACKEND": "tesseract",
                "VISION_LOCAL_CONTEXT_TESSERACT_PSM": "12",
                "VISION_LOCAL_CONTEXT_TESSERACT_LANG": "eng+chi_sim",
                "VISION_LOCAL_CONTEXT_CAPTION_BLOCKING": "1",
                "VISION_LOCAL_CONTEXT_CAPTION_ALLOW_DOWNLOAD": "true",
            },
            clear=True,
        ):
            self.assertEqual(vision_local_module._ocr_timeout_seconds(), 11)
            self.assertEqual(vision_local_module._ocr_backend_preference(), "tesseract")
            self.assertEqual(vision_local_module._tesseract_psm(), 12)
            self.assertEqual(vision_local_module._tesseract_lang(), "eng+chi_sim")
            self.assertTrue(vision_local_module._caption_blocking_enabled())
            self.assertTrue(vision_local_module._caption_allow_download())

    def test_analyze_image_skips_caption_for_text_heavy_ocr(self):
        with patch.object(vision_local_module, "_decode_image", return_value=(_FakeImage(), b"raw")), patch.object(
            vision_local_module, "_cache_get", return_value=None
        ), patch.object(vision_local_module, "_cache_put"), patch.object(
            vision_local_module,
            "_load_caption_backend",
            return_value=None,
        ), patch.object(
            vision_local_module,
            "_run_local_ocr",
            return_value="Settings Network Bluetooth Display Battery Privacy Power mode Screen brightness",
        ), patch.object(
            vision_local_module, "_caption_image", return_value="caption"
        ) as caption_mock:
            result = vision_local_module.analyze_image("abc123", debug_write=lambda *_args: None)

        caption_mock.assert_not_called()
        self.assertIn("Settings Network Bluetooth", result["ocr_text"])
        self.assertEqual(result["caption"], "")
        self.assertIn("system settings screen", result["summary"])
        self.assertIn("Readable labels include", result["summary"])

    def test_analyze_image_retries_ocr_when_initial_result_is_low_signal(self):
        with patch.object(vision_local_module, "_decode_image", return_value=(_FakeImage(), b"raw")), patch.object(
            vision_local_module, "_cache_get", return_value=None
        ), patch.object(vision_local_module, "_cache_put"), patch.object(
            vision_local_module,
            "_load_caption_backend",
            return_value=None,
        ), patch.object(
            vision_local_module,
            "_run_local_ocr",
            side_effect=[
                "Netmrk glueb:oth Display mo",
                "Settings Network Bluetooth Display Battery Privacy Battery Power mode Screen brightness",
            ],
        ) as ocr_mock, patch.object(
            vision_local_module,
            "_build_ocr_retry_image",
            return_value=_FakeImage(),
        ), patch.object(
            vision_local_module, "_caption_image", return_value=""
        ), patch.object(
            vision_local_module,
            "_get_caption_backend_state",
            return_value={"ready": False, "loading": False, "error": True},
        ):
            result = vision_local_module.analyze_image("abc123", debug_write=lambda *_args: None)

        self.assertEqual(ocr_mock.call_count, 2)
        self.assertIn("Settings", result["ocr_text"])
        self.assertIn("Network", result["summary"])

    def test_analyze_image_attempts_caption_for_ui_screens_even_with_some_ocr(self):
        with patch.object(vision_local_module, "_decode_image", return_value=(_FakeImage(), b"raw")), patch.object(
            vision_local_module, "_cache_get", return_value=None
        ), patch.object(vision_local_module, "_cache_put"), patch.object(
            vision_local_module,
            "_load_caption_backend",
            return_value=None,
        ), patch.object(
            vision_local_module,
            "_run_local_ocr",
            return_value="Settings Battery Power mode Screen brightness",
        ), patch.object(
            vision_local_module,
            "_build_ocr_retry_image",
            return_value=_FakeImage(),
        ), patch.object(
            vision_local_module,
            "_caption_image",
            return_value="a computer settings screen showing the user s settings",
        ) as caption_mock:
            result = vision_local_module.analyze_image("abc123", debug_write=lambda *_args: None)

        caption_mock.assert_called_once()
        self.assertEqual(
            caption_mock.call_args.kwargs["prompt"],
            "a computer settings screen showing",
        )
        self.assertIn("system settings screen", result["summary"])
        self.assertIn("Battery", result["summary"])

    def test_analyze_image_infers_chart_scene_and_summary_from_ocr(self):
        with patch.object(vision_local_module, "_decode_image", return_value=(_FakeImage(), b"raw")), patch.object(
            vision_local_module, "_cache_get", return_value=None
        ), patch.object(vision_local_module, "_cache_put"), patch.object(
            vision_local_module,
            "_load_caption_backend",
            return_value=None,
        ), patch.object(
            vision_local_module,
            "_run_local_ocr",
            return_value="Revenue Dashboard Q1 Q2 Q3 Q4 80k 60k 40k 20k Growth +18%",
        ), patch.object(
            vision_local_module,
            "_build_ocr_retry_image",
            return_value=_FakeImage(),
        ), patch.object(
            vision_local_module,
            "_analyze_chart_visual_pattern",
            return_value={"chart_like": True, "kind": "line", "trend": "upward", "series_count": 1},
        ), patch.object(
            vision_local_module,
            "_caption_image",
            return_value="",
        ), patch.object(
            vision_local_module,
            "_get_caption_backend_state",
            return_value={"ready": True, "loading": False, "error": False},
        ):
            result = vision_local_module.analyze_image("abc123", debug_write=lambda *_args: None)

        self.assertEqual(result["scene"], "chart")
        self.assertIn("line chart", result["summary"])
        self.assertIn("upward", result["summary"])
        self.assertIn("+18%", result["summary"])

    def test_analyze_chart_visual_pattern_detects_upward_line_chart(self):
        result = vision_local_module._analyze_chart_visual_pattern(_make_line_chart_image())

        self.assertTrue(result.get("chart_like"))
        self.assertEqual(result.get("kind"), "line")
        self.assertEqual(result.get("trend"), "upward")

    def test_analyze_chart_visual_pattern_detects_upward_bar_chart(self):
        result = vision_local_module._analyze_chart_visual_pattern(_make_bar_chart_image())

        self.assertTrue(result.get("chart_like"))
        self.assertEqual(result.get("kind"), "bar")
        self.assertEqual(result.get("trend"), "upward")

    def test_analyze_chart_visual_pattern_ignores_settings_controls(self):
        result = vision_local_module._analyze_chart_visual_pattern(_make_settings_like_image())

        self.assertEqual(result, {})

    def test_analyze_structured_layout_detects_browser_page(self):
        image = Image.new("RGB", (1280, 720), "white")
        lines = vision_local_module._normalize_ocr_lines(image, _browser_layout_lines())

        result = vision_local_module._analyze_structured_layout(
            image,
            lines,
            "https://example.com/login Example Login Email Password Sign in",
        )

        self.assertEqual(result.get("kind"), "browser")
        self.assertIn("example.com", result.get("address_bar", ""))
        self.assertIn("Example Login", result.get("page_title", ""))
        self.assertIn("Email", result.get("field_labels", []))

    def test_analyze_structured_layout_detects_chat_window(self):
        image = Image.new("RGB", (1280, 720), "white")
        lines = vision_local_module._normalize_ocr_lines(image, _chat_layout_lines())

        result = vision_local_module._analyze_structured_layout(
            image,
            lines,
            "Alice General Design Can you review this Sure send it over Type a message",
        )

        self.assertEqual(result.get("kind"), "chat")
        self.assertEqual(result.get("title"), "Alice")
        self.assertIn("Can you review this?", result.get("left_messages", []))
        self.assertIn("Sure, send it over", result.get("right_messages", []))
        self.assertEqual(result.get("input_hint"), "Type a message")

    def test_extract_chart_text_structure_collects_axes_and_annotations(self):
        image = Image.new("RGB", (1280, 800), "white")
        lines = vision_local_module._normalize_ocr_lines(image, _chart_axis_lines())

        result = vision_local_module._extract_chart_text_structure(image, lines)

        self.assertEqual(result.get("title"), "Revenue Dashboard")
        self.assertEqual(result.get("x_axis_labels"), ["Q1", "Q2", "Q3", "Q4"])
        self.assertEqual(result.get("y_axis_labels"), ["80k", "60k", "40k", "20k"])
        self.assertIn("Growth +18%", result.get("legend_labels", []))

    def test_merge_ocr_lines_prefers_higher_quality_retry_text_and_keeps_new_lines(self):
        primary = [
            {"text": "Example LCOin", "x": 460, "y": 142, "width": 67, "height": 10, "right": 527, "bottom": 152},
            {"text": "Email", "x": 421, "y": 242, "width": 25, "height": 8, "right": 446, "bottom": 250},
        ]
        secondary = [
            {"text": "ExampIe l-ogln", "x": 460, "y": 142, "width": 68, "height": 10, "right": 528, "bottom": 152},
            {"text": "Sign in", "x": 490, "y": 488, "width": 33, "height": 11, "right": 523, "bottom": 499},
        ]

        result = vision_local_module._merge_ocr_lines(primary, secondary)

        texts = [item["text"] for item in result]
        self.assertIn("ExampIe l-ogln", texts)
        self.assertIn("Sign in", texts)
        self.assertNotIn("Example LCOin", texts)

    def test_repair_short_ui_text_normalizes_common_ocr_mistakes(self):
        self.assertEqual(
            vision_local_module._repair_short_ui_text("ExampIe l-ogln"),
            "Example Login",
        )
        self.assertEqual(
            vision_local_module._repair_short_ui_text("Type a mesage"),
            "Type a message",
        )
        self.assertEqual(
            vision_local_module._repair_short_ui_text("https//example.com/logln", allow_url=True),
            "https://example.com/login",
        )

    def test_browser_layout_uses_upper_page_title_not_just_top_bar(self):
        image = Image.new("RGB", (1280, 720), "white")
        lines = vision_local_module._normalize_ocr_lines(
            image,
            [
                {"text": "https://example.com/login", "x": 160, "y": 26, "width": 540, "height": 24},
                {"text": "Example Login", "x": 460, "y": 182, "width": 220, "height": 24},
                {"text": "Email", "x": 420, "y": 240, "width": 80, "height": 20},
                {"text": "Sign in", "x": 490, "y": 486, "width": 90, "height": 20},
            ],
        )

        result = vision_local_module._analyze_structured_layout(
            image,
            lines,
            "https://example.com/login Example Login Email Sign in",
        )

        self.assertEqual(result.get("kind"), "browser")
        self.assertEqual(result.get("page_title"), "Example Login")

    def test_analyze_structured_layout_normalizes_noisy_browser_text(self):
        image = Image.new("RGB", (1280, 720), "white")
        lines = vision_local_module._normalize_ocr_lines(
            image,
            [
                {"text": "https//example.com/logln", "x": 160, "y": 26, "width": 540, "height": 24},
                {"text": "ExampIe l-ogln", "x": 460, "y": 182, "width": 220, "height": 24},
                {"text": "Email", "x": 420, "y": 240, "width": 80, "height": 20},
                {"text": "Sign in", "x": 490, "y": 486, "width": 90, "height": 20},
            ],
        )

        result = vision_local_module._analyze_structured_layout(
            image,
            lines,
            "https//example.com/logln ExampIe l-ogln Email Sign in",
        )

        self.assertEqual(result.get("kind"), "browser")
        self.assertEqual(result.get("address_bar"), "https://example.com/login")
        self.assertEqual(result.get("page_title"), "Example Login")

    def test_analyze_structured_layout_normalizes_noisy_chat_labels(self):
        image = Image.new("RGB", (1280, 720), "white")
        lines = vision_local_module._normalize_ocr_lines(
            image,
            [
                {"text": "Alice", "x": 560, "y": 38, "width": 80, "height": 24},
                {"text": "General", "x": 40, "y": 120, "width": 80, "height": 18},
                {"text": "Des gn", "x": 40, "y": 190, "width": 70, "height": 18},
                {"text": "Can you review this?", "x": 120, "y": 210, "width": 220, "height": 20},
                {"text": "Sure, send it over", "x": 760, "y": 280, "width": 220, "height": 20},
                {"text": "Type a mesage", "x": 420, "y": 660, "width": 180, "height": 20},
            ],
        )

        result = vision_local_module._analyze_structured_layout(
            image,
            lines,
            "Alice General Des gn Can you review this Sure send it over Type a mesage",
        )

        self.assertEqual(result.get("kind"), "chat")
        self.assertEqual(result.get("sidebar_labels"), ["General", "Design"])
        self.assertEqual(result.get("input_hint"), "Type a message")

    def test_analyze_image_uses_clean_visible_text_for_structured_browser_layout(self):
        ocr_payload = {
            "text": "https//example.com/logln ExampIe l-ogln Email Sign in",
            "lines": [
                {"text": "https//example.com/logln", "x": 160, "y": 26, "width": 540, "height": 24},
                {"text": "ExampIe l-ogln", "x": 460, "y": 182, "width": 220, "height": 24},
                {"text": "Email", "x": 420, "y": 240, "width": 80, "height": 20},
                {"text": "Sign in", "x": 490, "y": 486, "width": 90, "height": 20},
            ],
        }

        with patch.object(vision_local_module, "_decode_image", return_value=(_FakeImage(), b"raw")), patch.object(
            vision_local_module, "_cache_get", return_value=None
        ), patch.object(vision_local_module, "_cache_put"), patch.object(
            vision_local_module,
            "_load_caption_backend",
            return_value=None,
        ), patch.object(
            vision_local_module,
            "_run_local_ocr",
            return_value=ocr_payload,
        ), patch.object(
            vision_local_module,
            "_should_retry_ocr",
            return_value=False,
        ), patch.object(
            vision_local_module,
            "_should_attempt_caption",
            return_value=False,
        ), patch.object(
            vision_local_module,
            "_analyze_chart_visual_pattern",
            return_value={},
        ):
            result = vision_local_module.analyze_image("abc123", debug_write=lambda *_args: None)

        self.assertEqual(result["layout"]["address_bar"], "https://example.com/login")
        self.assertEqual(result["layout"]["page_title"], "Example Login")
        self.assertIn("https://example.com/login", result["visible_text"])
        self.assertIn("Example Login", result["visible_text"])
        self.assertNotIn("https//example.com/logln", result["visible_text"])

    def test_analyze_image_does_not_cache_caption_pending_result(self):
        with patch.object(vision_local_module, "_decode_image", return_value=(_FakeImage(), b"raw")), patch.object(
            vision_local_module, "_cache_get", return_value=None
        ), patch.object(vision_local_module, "_cache_put") as cache_put_mock, patch.object(
            vision_local_module,
            "_load_caption_backend",
            return_value=None,
        ), patch.object(
            vision_local_module,
            "_run_local_ocr",
            return_value="Settings Battery Power mode",
        ), patch.object(
            vision_local_module,
            "_build_ocr_retry_image",
            return_value=_FakeImage(),
        ), patch.object(
            vision_local_module, "_caption_image", return_value=""
        ), patch.object(
            vision_local_module,
            "_get_caption_backend_state",
            return_value={"ready": False, "loading": True, "error": False},
        ):
            result = vision_local_module.analyze_image("abc123", debug_write=lambda *_args: None)

        cache_put_mock.assert_not_called()
        self.assertTrue(result["caption_pending"])

    def test_build_user_image_context_includes_scene_type(self):
        with patch.object(
            vision_local_module,
            "analyze_image",
            return_value={
                "summary": "This looks like a dashboard or chart with numeric metrics.",
                "ocr_text": "Revenue Dashboard Q1 Q2 Q3 Q4 80k",
                "size": "1280x800",
                "scene": "chart",
                "chart_text": {
                    "title": "Revenue Dashboard",
                    "x_axis_labels": ["Q1", "Q2", "Q3", "Q4"],
                    "y_axis_labels": ["80k", "60k", "40k", "20k"],
                    "legend_labels": ["Growth +18%"],
                },
            },
        ):
            result = vision_local_module.build_user_image_context(["abc123"], user_text="describe it")

        self.assertIn("- Scene type: chart", result)
        self.assertIn("dashboard or chart", result)
        self.assertIn("- Chart details: title: Revenue Dashboard", result)

    def test_build_user_image_context_includes_layout_details(self):
        with patch.object(
            vision_local_module,
            "analyze_image",
            return_value={
                "summary": "This appears to be a browser or website page.",
                "ocr_text": "https://example.com/login Example Login Email Password",
                "size": "1280x720",
                "scene": "browser",
                "layout": {
                    "kind": "browser",
                    "address_bar": "https://example.com/login",
                    "page_title": "Example Login",
                    "field_labels": ["Email", "Password", "Sign in"],
                },
            },
        ):
            result = vision_local_module.build_user_image_context(["abc123"], user_text="describe it")

        self.assertIn("- Layout details: page title: Example Login", result)
        self.assertIn("address bar: https://example.com/login", result)

    def test_caption_image_defers_backend_loading_by_default(self):
        with patch.object(vision_local_module, "_CAPTION_BACKEND", None), patch.object(
            vision_local_module, "_CAPTION_LOAD_ATTEMPTED", False
        ), patch.object(
            vision_local_module, "_CAPTION_LOAD_ERROR", ""
        ), patch.object(
            vision_local_module, "_CAPTION_LOADING", False
        ), patch.object(
            vision_local_module, "_ensure_caption_backend_loading"
        ) as ensure_mock:
            result = vision_local_module._caption_image(_FakeImage(), lambda *_args: None)

        ensure_mock.assert_called_once()
        self.assertEqual(result, "")

    def test_run_windows_ocr_embeds_image_path_into_powershell_command(self):
        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return SimpleNamespace(returncode=0, stdout="hello", stderr="")

        with patch.object(vision_local_module.os, "name", "nt"), patch.object(
            vision_local_module.shutil, "which", return_value="powershell"
        ), patch.object(
            vision_local_module.tempfile,
            "NamedTemporaryFile",
            return_value=_FakeTempFile(r"C:\tmp\ocr test.png"),
        ), patch.object(
            vision_local_module.subprocess, "run", side_effect=fake_run
        ), patch.object(
            vision_local_module.os, "remove"
        ):
            result = vision_local_module._run_windows_ocr(_FakeImage(), lambda *_args: None)

        self.assertEqual(result, "hello")
        self.assertEqual(captured["args"][:3], ["powershell", "-NoProfile", "-Command"])
        self.assertEqual(len(captured["args"]), 4)
        self.assertNotIn("__IMAGE_PATH__", captured["args"][3])
        self.assertIn(r"C:\tmp\ocr test.png", captured["args"][3])

    def test_run_windows_ocr_include_layout_parses_json_payload(self):
        payload = {
            "text": "Example Login Email Password",
            "lines": [
                {"text": "Example Login", "x": 220, "y": 118, "width": 220, "height": 24},
                {"text": "Email", "x": 230, "y": 250, "width": 80, "height": 20},
            ],
        }

        def fake_run(_args, **_kwargs):
            return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

        with patch.object(vision_local_module.os, "name", "nt"), patch.object(
            vision_local_module.shutil, "which", return_value="powershell"
        ), patch.object(
            vision_local_module.tempfile,
            "NamedTemporaryFile",
            return_value=_FakeTempFile(r"C:\tmp\ocr test.png"),
        ), patch.object(
            vision_local_module.subprocess, "run", side_effect=fake_run
        ), patch.object(
            vision_local_module.os, "remove"
        ):
            result = vision_local_module._run_windows_ocr(
                _FakeImage(),
                lambda *_args: None,
                include_layout=True,
            )

        self.assertEqual(result["text"], "Example Login Email Password")
        self.assertEqual(len(result["lines"]), 2)
        self.assertEqual(result["lines"][0]["text"], "Example Login")

    def test_run_tesseract_ocr_include_layout_parses_tsv_payload(self):
        tsv_payload = "\n".join(
            [
                "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
                "4\t1\t1\t1\t1\t0\t220\t118\t220\t24\t-1\t",
                "5\t1\t1\t1\t1\t1\t220\t118\t92\t24\t96.0\tExample",
                "5\t1\t1\t1\t1\t2\t324\t118\t116\t24\t95.0\tLogin",
                "4\t1\t1\t1\t2\t0\t230\t250\t80\t20\t-1\t",
                "5\t1\t1\t1\t2\t1\t230\t250\t80\t20\t93.0\tEmail",
            ]
        )

        def fake_run(_args, **_kwargs):
            return SimpleNamespace(returncode=0, stdout=tsv_payload, stderr="")

        with patch.object(
            vision_local_module.shutil,
            "which",
            side_effect=lambda name: "tesseract" if name == "tesseract" else None,
        ), patch.object(
            vision_local_module.tempfile,
            "NamedTemporaryFile",
            return_value=_FakeTempFile(r"C:\tmp\ocr test.png"),
        ), patch.object(
            vision_local_module.subprocess, "run", side_effect=fake_run
        ), patch.object(
            vision_local_module.os, "remove"
        ):
            result = vision_local_module._run_tesseract_ocr(
                _FakeImage(),
                lambda *_args: None,
                include_layout=True,
            )

        self.assertEqual(result["text"], "Example Login Email")
        self.assertEqual(len(result["lines"]), 2)
        self.assertEqual(result["lines"][0]["text"], "Example Login")
        self.assertEqual(result["lines"][1]["text"], "Email")

    def test_run_tesseract_ocr_honors_lang_and_psm_configuration(self):
        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return SimpleNamespace(returncode=0, stdout="hello", stderr="")

        with patch.dict(
            vision_local_module.os.environ,
            {
                "VISION_LOCAL_CONTEXT_TESSERACT_LANG": "eng+chi_sim",
                "VISION_LOCAL_CONTEXT_TESSERACT_PSM": "12",
            },
            clear=False,
        ), patch.object(
            vision_local_module.shutil,
            "which",
            side_effect=lambda name: "tesseract" if name == "tesseract" else None,
        ), patch.object(
            vision_local_module.tempfile,
            "NamedTemporaryFile",
            return_value=_FakeTempFile(r"C:\tmp\ocr test.png"),
        ), patch.object(
            vision_local_module.subprocess, "run", side_effect=fake_run
        ), patch.object(
            vision_local_module.os, "remove"
        ):
            result = vision_local_module._run_tesseract_ocr(_FakeImage(), lambda *_args: None)

        self.assertEqual(result, "hello")
        self.assertEqual(
            captured["args"],
            ["tesseract", r"C:\tmp\ocr test.png", "-", "--psm", "12", "-l", "eng+chi_sim", "quiet"],
        )

    def test_run_local_ocr_uses_tesseract_when_windows_is_unavailable(self):
        with patch.object(vision_local_module, "has_windows_ocr_support", return_value=False), patch.object(
            vision_local_module, "has_tesseract_ocr_support", return_value=True
        ), patch.object(
            vision_local_module, "_run_tesseract_ocr", return_value="hello"
        ) as tesseract_mock, patch.object(
            vision_local_module, "_run_windows_ocr"
        ) as windows_mock:
            result = vision_local_module._run_local_ocr(_FakeImage(), lambda *_args: None)

        windows_mock.assert_not_called()
        tesseract_mock.assert_called_once()
        self.assertEqual(result, "hello")


if __name__ == "__main__":
    unittest.main()
