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
        image = Image.new("RGB", (1365, 768), "white")
        ocr_payload = {
            "text": "https://example.cn/login 示例 Login 邮箱 密码 登录",
            "lines": [
                {"text": "https://example.cn/login", "x": 150, "y": 26, "width": 620, "height": 24},
                {"text": "示例 Login", "x": 470, "y": 164, "width": 220, "height": 28},
                {"text": "邮箱", "x": 430, "y": 264, "width": 52, "height": 20},
                {"text": "密码", "x": 430, "y": 332, "width": 52, "height": 20},
                {"text": "登录", "x": 518, "y": 468, "width": 52, "height": 20},
            ],
        }

        result = self._analyze_with_payload(image, ocr_payload, chart_visual={})

        self.assertEqual(result["scene"], "browser")
        self.assertEqual(result["layout"]["kind"], "browser")
        self.assertEqual(result["layout"]["address_bar"], "https://example.cn/login")
        self.assertEqual(result["layout"]["page_title"], "示例 Login")
        self.assertEqual(result["layout"]["field_labels"], ["邮箱", "密码", "登录"])
        self.assertIn("browser or website page", result["summary"])
        self.assertIn("示例 Login", result["visible_text"])
        self.assertIn("邮箱", result["visible_text"])

    def test_golden_low_res_mixed_language_chat_screen(self):
        image = Image.new("RGB", (640, 360), "white")
        ocr_payload = {
            "text": "产品群 产品 Design 请 review 一下 PR 我看过了 稍后回复 输入消息 发送 消息 聊天",
            "lines": [
                {"text": "产品群", "x": 286, "y": 22, "width": 68, "height": 18},
                {"text": "产品", "x": 28, "y": 88, "width": 44, "height": 18},
                {"text": "Design", "x": 28, "y": 122, "width": 68, "height": 18},
                {"text": "请 review 一下 PR", "x": 96, "y": 146, "width": 170, "height": 18},
                {"text": "我看过了，稍后回复", "x": 380, "y": 188, "width": 146, "height": 18},
                {"text": "输入消息", "x": 236, "y": 322, "width": 86, "height": 18},
                {"text": "发送", "x": 544, "y": 322, "width": 42, "height": 18},
            ],
        }

        result = self._analyze_with_payload(image, ocr_payload, chart_visual={})

        self.assertEqual(result["scene"], "chat")
        self.assertEqual(result["layout"]["kind"], "chat")
        self.assertEqual(result["layout"]["title"], "产品群")
        self.assertEqual(result["layout"]["input_hint"], "输入消息")
        self.assertEqual(result["layout"]["sidebar_labels"], ["产品", "Design"])
        self.assertIn("请 review 一下 PR", result["layout"]["left_messages"])
        self.assertIn("我看过了，稍后回复", result["layout"]["right_messages"])
        self.assertIn("chat or messaging interface", result["summary"])
        self.assertIn("输入消息", result["visible_text"])

    def test_golden_dashboard_chart_edge_case_detects_downward_trend(self):
        image = _make_downward_bar_chart_image()
        ocr_payload = {
            "text": "季度营收 Dashboard Q1 Q2 Q3 Q4 80k 60k 40k 20k 同比 -12%",
            "lines": [
                {"text": "季度营收 Dashboard", "x": 80, "y": 70, "width": 280, "height": 24},
                {"text": "80k", "x": 60, "y": 250, "width": 40, "height": 18},
                {"text": "60k", "x": 60, "y": 360, "width": 40, "height": 18},
                {"text": "40k", "x": 60, "y": 470, "width": 40, "height": 18},
                {"text": "20k", "x": 60, "y": 580, "width": 40, "height": 18},
                {"text": "Q1", "x": 260, "y": 660, "width": 30, "height": 18},
                {"text": "Q2", "x": 470, "y": 660, "width": 30, "height": 18},
                {"text": "Q3", "x": 680, "y": 660, "width": 30, "height": 18},
                {"text": "Q4", "x": 890, "y": 660, "width": 30, "height": 18},
                {"text": "同比 -12%", "x": 980, "y": 135, "width": 110, "height": 18},
            ],
        }

        result = self._analyze_with_payload(image, ocr_payload)

        self.assertEqual(result["scene"], "chart")
        self.assertEqual(result["chart_visual"]["kind"], "bar")
        self.assertEqual(result["chart_visual"]["trend"], "downward")
        self.assertEqual(result["chart_text"]["title"], "季度营收 Dashboard")
        self.assertEqual(result["chart_text"]["x_axis_labels"], ["Q1", "Q2", "Q3", "Q4"])
        self.assertIn("同比 -12%", result["chart_text"]["legend_labels"])
        self.assertIn("bar chart", result["summary"])
        self.assertIn("downward", result["summary"])


if __name__ == "__main__":
    unittest.main()
