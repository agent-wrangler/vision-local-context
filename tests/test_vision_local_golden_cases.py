import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import vision_local as vision_local_module


_USE_REAL_CHART_VISUAL = object()


def _make_downward_bar_chart_image():
    image = Image.new("RGB", (1280, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((40, 40, 1240, 760), radius=18, fill="#ffffff", outline="#d1d5db", width=2)
    draw.line((120, 640, 1160, 640), fill="#9ca3af", width=3)
    draw.line((120, 180, 120, 640), fill="#9ca3af", width=3)
    for x, top in [(220, 220), (450, 300), (680, 400), (910, 500)]:
        draw.rounded_rectangle((x, top, x + 90, 640), radius=8, fill="#ef4444")
    return image


def _make_downward_line_chart_image():
    image = Image.new("RGB", (1280, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((40, 40, 1240, 760), radius=18, fill="#ffffff", outline="#d1d5db", width=2)
    draw.line((120, 640, 1160, 640), fill="#9ca3af", width=3)
    draw.line((120, 180, 120, 640), fill="#9ca3af", width=3)
    points = [(260, 250), (470, 310), (680, 420), (890, 560)]
    draw.line(points, fill="#2563eb", width=6)
    for x, y in points:
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill="#2563eb")
    return image


def _make_low_res_browser_image():
    image = Image.new("RGB", (720, 405), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((16, 16, 704, 389), radius=14, fill="#ffffff", outline="#cbd5e1", width=2)
    draw.rounded_rectangle((52, 42, 626, 74), radius=10, fill="#eef2ff", outline="#cbd5e1", width=2)
    draw.rounded_rectangle((212, 152, 508, 188), radius=8, outline="#94a3b8", width=2)
    draw.rounded_rectangle((212, 218, 508, 254), radius=8, outline="#94a3b8", width=2)
    draw.rounded_rectangle((282, 288, 438, 326), radius=8, fill="#2563eb")
    return image


def _make_settings_panel_image():
    image = Image.new("RGB", (1280, 800), "#f3f4f6")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((40, 40, 1240, 760), radius=18, fill="#ffffff", outline="#d1d5db", width=2)
    draw.rectangle((40, 40, 1240, 110), fill="#111827")
    for y in (188, 246, 304, 362, 420):
        draw.rounded_rectangle((92, y, 300, y + 34), radius=10, fill="#f8fafc", outline="#e5e7eb", width=1)
    draw.rounded_rectangle((720, 214, 930, 254), radius=20, fill="#dbeafe")
    draw.rounded_rectangle((720, 300, 820, 348), radius=24, fill="#22c55e")
    draw.rectangle((720, 410, 1040, 418), fill="#d1d5db")
    draw.rectangle((720, 410, 920, 418), fill="#3b82f6")
    draw.ellipse((910, 398, 934, 422), fill="#2563eb")
    return image


def _make_low_res_chat_image():
    image = Image.new("RGB", (640, 360), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((10, 10, 630, 350), radius=12, fill="#ffffff", outline="#cbd5e1", width=2)
    draw.rectangle((10, 10, 144, 350), fill="#f1f5f9")
    draw.rounded_rectangle((82, 138, 286, 172), radius=10, fill="#e2e8f0")
    draw.rounded_rectangle((356, 180, 560, 214), radius=10, fill="#dbeafe")
    draw.rounded_rectangle((206, 310, 420, 340), radius=10, fill="#f8fafc", outline="#cbd5e1", width=1)
    return image


class VisionLocalGoldenRegressionTests(unittest.TestCase):
    def _analyze_with_payload(
        self,
        image: Image.Image,
        ocr_payload,
        *,
        chart_visual=_USE_REAL_CHART_VISUAL,
        should_attempt_caption: bool = False,
        caption: str = "",
        caption_state: dict | None = None,
    ) -> dict:
        caption_state = caption_state or {"ready": False, "loading": False, "error": False}
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    vision_local_module,
                    "_decode_image",
                    return_value=(image, b"golden-raw"),
                )
            )
            stack.enter_context(patch.object(vision_local_module, "_cache_get", return_value=None))
            stack.enter_context(patch.object(vision_local_module, "_cache_put"))
            stack.enter_context(patch.object(vision_local_module, "_load_caption_backend", return_value=None))
            stack.enter_context(
                patch.object(
                    vision_local_module,
                    "_run_local_ocr_with_backend",
                    return_value=(ocr_payload, "windows"),
                )
            )
            stack.enter_context(patch.object(vision_local_module, "_should_retry_ocr", return_value=False))
            stack.enter_context(
                patch.object(
                    vision_local_module,
                    "_should_attempt_caption",
                    return_value=should_attempt_caption,
                )
            )
            stack.enter_context(patch.object(vision_local_module, "_caption_image", return_value=caption))
            stack.enter_context(
                patch.object(
                    vision_local_module,
                    "_get_caption_backend_state",
                    return_value=caption_state,
                )
            )
            if chart_visual is not _USE_REAL_CHART_VISUAL:
                stack.enter_context(
                    patch.object(
                        vision_local_module,
                        "_analyze_chart_visual_pattern",
                        return_value=chart_visual,
                    )
                )
            return vision_local_module.analyze_image("golden-case", debug_write=lambda *_args: None)

    def test_golden_mixed_language_browser_login_screen(self):
        title = "\u793a\u4f8b Login"
        fields = ["\u90ae\u7bb1", "\u5bc6\u7801", "\u767b\u5f55"]
        image = Image.new("RGB", (1365, 768), "white")
        ocr_payload = {
            "text": f"https://example.cn/login {title} {' '.join(fields)}",
            "lines": [
                {"text": "https://example.cn/login", "x": 150, "y": 26, "width": 620, "height": 24},
                {"text": title, "x": 470, "y": 164, "width": 220, "height": 28},
                {"text": fields[0], "x": 430, "y": 264, "width": 52, "height": 20},
                {"text": fields[1], "x": 430, "y": 332, "width": 52, "height": 20},
                {"text": fields[2], "x": 518, "y": 468, "width": 52, "height": 20},
            ],
        }

        result = self._analyze_with_payload(image, ocr_payload, chart_visual={})

        self.assertEqual(result["scene"], "browser")
        self.assertEqual(result["layout"]["kind"], "browser")
        self.assertEqual(result["layout"]["address_bar"], "https://example.cn/login")
        self.assertEqual(result["layout"]["page_title"], title)
        self.assertEqual(result["layout"]["field_labels"], fields)
        self.assertIn("browser or website page", result["summary"])
        self.assertIn(title, result["visible_text"])
        self.assertIn(fields[0], result["visible_text"])

    def test_golden_browser_preserves_bilingual_field_labels(self):
        title = "\u793a\u4f8b Login"
        fields = ["\u90ae\u7bb1 / Email", "\u5bc6\u7801 / Password", "\u767b\u5f55 Sign in"]
        image = Image.new("RGB", (1365, 768), "white")
        ocr_payload = {
            "text": f"https://example.cn/login {title} {' '.join(fields)}",
            "lines": [
                {"text": "https://example.cn/login", "x": 150, "y": 26, "width": 620, "height": 24},
                {"text": title, "x": 470, "y": 164, "width": 220, "height": 28},
                {"text": fields[0], "x": 430, "y": 264, "width": 140, "height": 20},
                {"text": fields[1], "x": 430, "y": 332, "width": 160, "height": 20},
                {"text": fields[2], "x": 518, "y": 468, "width": 120, "height": 20},
            ],
        }

        result = self._analyze_with_payload(image, ocr_payload, chart_visual={})

        self.assertEqual(result["scene"], "browser")
        self.assertEqual(result["layout"]["field_labels"], fields)
        self.assertIn(fields[1], result["visible_text"])
        self.assertIn(fields[2], result["visible_text"])

    def test_golden_low_res_mixed_language_chat_screen(self):
        title = "\u4ea7\u54c1\u7fa4"
        sidebar = ["\u4ea7\u54c1", "Design"]
        left_message = "\u8bf7 review \u4e00\u4e0b PR"
        right_message = "\u6211\u770b\u8fc7\u4e86\uff0c\u7a0d\u540e\u56de\u590d"
        input_hint = "\u8f93\u5165\u6d88\u606f"
        image = Image.new("RGB", (640, 360), "white")
        ocr_payload = {
            "text": f"{title} {' '.join(sidebar)} {left_message} {right_message} {input_hint} \u53d1\u9001 \u804a\u5929",
            "lines": [
                {"text": title, "x": 286, "y": 22, "width": 68, "height": 18},
                {"text": sidebar[0], "x": 28, "y": 88, "width": 44, "height": 18},
                {"text": sidebar[1], "x": 28, "y": 122, "width": 68, "height": 18},
                {"text": left_message, "x": 96, "y": 146, "width": 170, "height": 18},
                {"text": right_message, "x": 380, "y": 188, "width": 146, "height": 18},
                {"text": input_hint, "x": 236, "y": 322, "width": 86, "height": 18},
                {"text": "\u53d1\u9001", "x": 544, "y": 322, "width": 42, "height": 18},
            ],
        }

        result = self._analyze_with_payload(image, ocr_payload, chart_visual={})

        self.assertEqual(result["scene"], "chat")
        self.assertEqual(result["layout"]["kind"], "chat")
        self.assertEqual(result["layout"]["title"], title)
        self.assertEqual(result["layout"]["input_hint"], input_hint)
        self.assertEqual(result["layout"]["sidebar_labels"], sidebar)
        self.assertIn(left_message, result["layout"]["left_messages"])
        self.assertIn(right_message, result["layout"]["right_messages"])
        self.assertIn("chat or messaging interface", result["summary"])
        self.assertIn(input_hint, result["visible_text"])

    def test_golden_dashboard_chart_edge_case_detects_downward_trend(self):
        title = "\u5b63\u5ea6\u8425\u6536 Dashboard"
        annotation = "\u540c\u6bd4 -12%"
        image = _make_downward_bar_chart_image()
        ocr_payload = {
            "text": f"{title} Q1 Q2 Q3 Q4 80k 60k 40k 20k {annotation}",
            "lines": [
                {"text": title, "x": 80, "y": 70, "width": 280, "height": 24},
                {"text": "80k", "x": 60, "y": 250, "width": 40, "height": 18},
                {"text": "60k", "x": 60, "y": 360, "width": 40, "height": 18},
                {"text": "40k", "x": 60, "y": 470, "width": 40, "height": 18},
                {"text": "20k", "x": 60, "y": 580, "width": 40, "height": 18},
                {"text": "Q1", "x": 260, "y": 660, "width": 30, "height": 18},
                {"text": "Q2", "x": 470, "y": 660, "width": 30, "height": 18},
                {"text": "Q3", "x": 680, "y": 660, "width": 30, "height": 18},
                {"text": "Q4", "x": 890, "y": 660, "width": 30, "height": 18},
                {"text": annotation, "x": 980, "y": 135, "width": 110, "height": 18},
            ],
        }

        result = self._analyze_with_payload(image, ocr_payload)

        self.assertEqual(result["scene"], "chart")
        self.assertEqual(result["chart_visual"]["kind"], "bar")
        self.assertEqual(result["chart_visual"]["trend"], "downward")
        self.assertEqual(result["chart_text"]["title"], title)
        self.assertEqual(result["chart_text"]["x_axis_labels"], ["Q1", "Q2", "Q3", "Q4"])
        self.assertIn(annotation, result["chart_text"]["legend_labels"])
        self.assertIn("bar chart", result["summary"])
        self.assertIn("downward", result["summary"])

    def test_golden_low_res_browser_repairs_url_and_keeps_mixed_fields(self):
        password_label = "\u5bc6\u7801"
        image = _make_low_res_browser_image()
        ocr_payload = {
            "text": f"https//example.cn/account/security Account Security Email {password_label} Sign in",
            "lines": [
                {"text": "https//example.cn/account/security", "x": 92, "y": 24, "width": 430, "height": 18},
                {"text": "Account Security", "x": 252, "y": 104, "width": 166, "height": 20},
                {"text": "Email", "x": 220, "y": 180, "width": 60, "height": 18},
                {"text": password_label, "x": 220, "y": 232, "width": 40, "height": 18},
                {"text": "Sign in", "x": 286, "y": 300, "width": 66, "height": 18},
            ],
        }

        result = self._analyze_with_payload(image, ocr_payload, chart_visual={})

        self.assertEqual(result["scene"], "browser")
        self.assertEqual(result["layout"]["kind"], "browser")
        self.assertEqual(result["layout"]["address_bar"], "https://example.cn/account/security")
        self.assertEqual(result["layout"]["page_title"], "Account Security")
        self.assertEqual(result["layout"]["field_labels"], ["Email", password_label, "Sign in"])
        self.assertIn("browser or website page", result["summary"])
        self.assertIn("https://example.cn/account/security", result["visible_text"])

    def test_golden_mixed_language_settings_screen_highlights_focus_areas(self):
        image = _make_settings_panel_image()
        ocr_payload = {
            "text": (
                "Settings \u8bbe\u7f6e Network \u7f51\u7edc Wi-Fi Bluetooth "
                "Privacy \u9690\u79c1 Battery \u7535\u6c60"
            ),
            "lines": [
                {"text": "Settings \u8bbe\u7f6e", "x": 110, "y": 74, "width": 160, "height": 24},
                {"text": "Network \u7f51\u7edc", "x": 118, "y": 196, "width": 150, "height": 22},
                {"text": "Wi-Fi", "x": 118, "y": 252, "width": 68, "height": 22},
                {"text": "Bluetooth", "x": 118, "y": 308, "width": 90, "height": 22},
                {"text": "Privacy \u9690\u79c1", "x": 118, "y": 364, "width": 126, "height": 22},
                {"text": "Battery \u7535\u6c60", "x": 118, "y": 420, "width": 132, "height": 22},
            ],
        }

        result = self._analyze_with_payload(image, ocr_payload, chart_visual={})

        self.assertEqual(result["scene"], "settings")
        self.assertEqual(result["layout"], {})
        self.assertIn("system settings screen", result["summary"])
        self.assertIn("connectivity", result["summary"])
        self.assertIn("privacy", result["summary"])
        self.assertIn("Settings", result["visible_text"])
        self.assertIn("Privacy", result["visible_text"])

    def test_golden_mixed_language_line_chart_extracts_axis_and_annotations(self):
        title = "Revenue \u8d8b\u52bf Dashboard"
        image = _make_downward_line_chart_image()
        ocr_payload = {
            "text": f"{title} Q1 Q2 Q3 Q4 80k 60k 40k 20k YoY -18% North America",
            "lines": [
                {"text": title, "x": 80, "y": 70, "width": 320, "height": 24},
                {"text": "80k", "x": 60, "y": 250, "width": 40, "height": 18},
                {"text": "60k", "x": 60, "y": 360, "width": 40, "height": 18},
                {"text": "40k", "x": 60, "y": 470, "width": 40, "height": 18},
                {"text": "20k", "x": 60, "y": 580, "width": 40, "height": 18},
                {"text": "Q1", "x": 260, "y": 660, "width": 30, "height": 18},
                {"text": "Q2", "x": 470, "y": 660, "width": 30, "height": 18},
                {"text": "Q3", "x": 680, "y": 660, "width": 30, "height": 18},
                {"text": "Q4", "x": 890, "y": 660, "width": 30, "height": 18},
                {"text": "YoY -18%", "x": 980, "y": 135, "width": 90, "height": 18},
                {"text": "North America", "x": 930, "y": 178, "width": 120, "height": 18},
            ],
        }

        result = self._analyze_with_payload(image, ocr_payload)

        self.assertEqual(result["scene"], "chart")
        self.assertEqual(result["chart_visual"]["kind"], "line")
        self.assertEqual(result["chart_visual"]["trend"], "downward")
        self.assertEqual(result["chart_text"]["title"], title)
        self.assertEqual(result["chart_text"]["x_axis_labels"], ["Q1", "Q2", "Q3", "Q4"])
        self.assertEqual(result["chart_text"]["y_axis_labels"], ["80k", "60k", "40k", "20k"])
        self.assertIn("YoY -18%", result["chart_text"]["legend_labels"])
        self.assertIn("North America", result["chart_text"]["legend_labels"])
        self.assertIn("line chart", result["summary"])
        self.assertIn("downward", result["summary"])
        self.assertIn("quarter", result["summary"])
        self.assertIn("YoY -18%", result["visible_text"])

    def test_golden_low_signal_ocr_retry_recovers_chat_layout(self):
        image = _make_low_res_chat_image()
        first_payload = (
            {
                "text": "abc 12",
                "lines": [
                    {"text": "abc 12", "x": 20, "y": 20, "width": 72, "height": 24},
                ],
            },
            "windows",
        )
        retry_payload = (
            {
                "text": (
                    "Team Chat Product Design Please review the PR before 5pm "
                    "Okay, I will reply after reading it Type a message Send"
                ),
                "lines": [
                    {"text": "Team Chat", "x": 524, "y": 44, "width": 180, "height": 32},
                    {"text": "Product", "x": 48, "y": 172, "width": 64, "height": 32},
                    {"text": "Design", "x": 48, "y": 240, "width": 104, "height": 32},
                    {
                        "text": "Please review the PR before 5pm",
                        "x": 176,
                        "y": 292,
                        "width": 380,
                        "height": 36,
                    },
                    {
                        "text": "Okay, I will reply after reading it",
                        "x": 752,
                        "y": 376,
                        "width": 340,
                        "height": 36,
                    },
                    {"text": "Type a message", "x": 456, "y": 644, "width": 192, "height": 36},
                    {"text": "Send", "x": 1088, "y": 644, "width": 64, "height": 36},
                ],
            },
            "tesseract",
        )

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    vision_local_module,
                    "_decode_image",
                    return_value=(image, b"golden-raw"),
                )
            )
            stack.enter_context(patch.object(vision_local_module, "_cache_get", return_value=None))
            stack.enter_context(patch.object(vision_local_module, "_cache_put"))
            stack.enter_context(patch.object(vision_local_module, "_load_caption_backend", return_value=None))
            stack.enter_context(
                patch.object(
                    vision_local_module,
                    "_run_local_ocr_with_backend",
                    side_effect=[first_payload, retry_payload],
                )
            )
            stack.enter_context(patch.object(vision_local_module, "_should_attempt_caption", return_value=False))
            stack.enter_context(patch.object(vision_local_module, "_caption_image", return_value=""))
            stack.enter_context(
                patch.object(
                    vision_local_module,
                    "_get_caption_backend_state",
                    return_value={"ready": False, "loading": False, "error": False},
                )
            )
            stack.enter_context(
                patch.object(
                    vision_local_module,
                    "_analyze_chart_visual_pattern",
                    return_value={},
                )
            )
            result = vision_local_module.analyze_image("golden-retry-case", debug_write=lambda *_args: None)

        self.assertEqual(result["scene"], "chat")
        self.assertEqual(result["ocr_backend"], "tesseract")
        self.assertEqual(result["layout"]["kind"], "chat")
        self.assertEqual(result["layout"]["title"], "Team Chat")
        self.assertEqual(result["layout"]["left_messages"], ["Please review the PR before 5pm"])
        self.assertEqual(result["layout"]["right_messages"], ["Okay, I will reply after reading it"])
        self.assertEqual(result["layout"]["input_hint"], "Type a message")
        self.assertIn("chat or messaging interface", result["summary"])
        self.assertIn("Type a message", result["visible_text"])


if __name__ == "__main__":
    unittest.main()
