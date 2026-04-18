from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parent
FIXTURES_DIR = ROOT / "fixtures"

FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/Deng.ttf",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _save(image: Image.Image, name: str) -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    image.save(FIXTURES_DIR / name, format="PNG", optimize=True)


def _draw_browser_fixture() -> Image.Image:
    image = Image.new("RGB", (1366, 820), "#eef2f7")
    draw = ImageDraw.Draw(image)
    title_font = _load_font(54)
    body_font = _load_font(34)
    label_font = _load_font(30)
    button_font = _load_font(34)
    helper_font = _load_font(24)

    draw.rounded_rectangle((52, 36, 1314, 784), radius=24, fill="#ffffff", outline="#cbd5e1", width=3)
    draw.rounded_rectangle((96, 70, 1240, 120), radius=14, fill="#f8fafc", outline="#cbd5e1", width=2)
    draw.text((132, 82), "https://example.cn/login", fill="#0f172a", font=body_font)

    draw.rounded_rectangle((392, 172, 972, 706), radius=26, fill="#ffffff", outline="#dbe3ee", width=2)
    draw.text((512, 228), "\u793a\u4f8b Login", fill="#111827", font=title_font)
    draw.text((472, 286), "\u7ee7\u7eed\u4f7f\u7528\u90ae\u7bb1\u767b\u5f55", fill="#475569", font=helper_font)

    draw.text((456, 354), "\u90ae\u7bb1 / Email", fill="#111827", font=label_font)
    draw.rounded_rectangle((448, 392, 916, 446), radius=12, outline="#94a3b8", width=2)
    draw.text((470, 404), "you@example.cn", fill="#64748b", font=helper_font)

    draw.text((456, 474), "\u5bc6\u7801 / Password", fill="#111827", font=label_font)
    draw.rounded_rectangle((448, 512, 916, 566), radius=12, outline="#94a3b8", width=2)
    draw.text((470, 524), "************", fill="#64748b", font=helper_font)

    draw.rounded_rectangle((520, 618, 844, 682), radius=14, fill="#2563eb")
    draw.text((590, 632), "\u767b\u5f55 Sign in", fill="#ffffff", font=button_font)
    return image


def _draw_chat_fixture() -> Image.Image:
    base = Image.new("RGB", (1280, 720), "#f4f7fb")
    draw = ImageDraw.Draw(base)
    title_font = _load_font(42)
    body_font = _load_font(30)
    small_font = _load_font(24)

    draw.rounded_rectangle((20, 20, 1260, 700), radius=24, fill="#ffffff", outline="#d2dae6", width=3)
    draw.rectangle((20, 20, 250, 700), fill="#f0f4f8")
    draw.rounded_rectangle((290, 34, 1236, 92), radius=18, fill="#ffffff", outline="#d7dfeb", width=1)
    draw.text((612, 44), "\u4ea7\u54c1\u7fa4 Product Sync", fill="#111827", font=title_font, anchor="ma")

    draw.text((56, 134), "\u4ea7\u54c1", fill="#0f172a", font=body_font)
    draw.text((56, 198), "Design", fill="#0f172a", font=body_font)
    draw.text((56, 262), "QA", fill="#0f172a", font=body_font)

    draw.rounded_rectangle((316, 182, 712, 264), radius=22, fill="#eef2ff")
    draw.text((344, 204), "\u8bf7 review \u4e00\u4e0b PR #204", fill="#111827", font=body_font)

    draw.rounded_rectangle((744, 304, 1092, 386), radius=22, fill="#dbeafe")
    draw.text((776, 326), "\u6211\u770b\u5b8c\u540e\u56de\u590d", fill="#0f172a", font=body_font)

    draw.rounded_rectangle((320, 612, 972, 668), radius=18, fill="#f8fafc", outline="#cbd5e1", width=2)
    draw.text((354, 628), "\u8f93\u5165\u6d88\u606f / Reply", fill="#475569", font=small_font)

    draw.rounded_rectangle((1002, 612, 1154, 668), radius=18, fill="#2563eb")
    draw.text((1078, 628), "\u53d1\u9001", fill="#ffffff", font=body_font, anchor="ma")

    low_res = base.resize((640, 360), Image.Resampling.LANCZOS)
    return low_res.filter(ImageFilter.SHARPEN)


def main() -> None:
    _save(_draw_browser_fixture(), "mixed_browser_login.png")
    _save(_draw_chat_fixture(), "mixed_chat_low_res.png")


if __name__ == "__main__":
    main()
