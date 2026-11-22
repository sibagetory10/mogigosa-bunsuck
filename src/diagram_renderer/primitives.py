from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from src.models import KoreanDiagramNode

_PALETTES: dict[str, dict[str, tuple[int, int, int]]] = {
    "blue": {
        "bg": (239, 246, 255),
        "border": (59, 130, 246),
        "hdr": (29, 78, 216),
        "badge_bg": (219, 234, 254),
        "badge_fg": (30, 64, 175),
    },
    "green": {
        "bg": (240, 253, 244),
        "border": (34, 197, 94),
        "hdr": (21, 128, 61),
        "badge_bg": (220, 252, 231),
        "badge_fg": (22, 101, 52),
    },
    "amber": {
        "bg": (255, 251, 235),
        "border": (245, 158, 11),
        "hdr": (180, 83, 9),
        "badge_bg": (254, 243, 199),
        "badge_fg": (146, 64, 14),
    },
    "purple": {
        "bg": (250, 245, 255),
        "border": (168, 85, 247),
        "hdr": (126, 34, 206),
        "badge_bg": (243, 232, 255),
        "badge_fg": (107, 33, 168),
    },
    "slate": {
        "bg": (248, 250, 252),
        "border": (100, 116, 139),
        "hdr": (51, 65, 85),
        "badge_bg": (226, 232, 240),
        "badge_fg": (30, 41, 59),
    },
}

_COLOR_CYCLE = ("blue", "green", "amber", "purple", "slate")

_REGULAR_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/NanumBarunGothic.ttf",
    "C:/Windows/Fonts/NanumGothic.ttf",
)

_BOLD_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/NanumBarunGothicBold.ttf",
    "C:/Windows/Fonts/NanumGothicBold.ttf",
    "C:/Windows/Fonts/malgun.ttf",
)


@lru_cache(maxsize=16)
def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = _BOLD_FONT_CANDIDATES if bold else _REGULAR_FONT_CANDIDATES
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    for path in _REGULAR_FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _measure_line_height(
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    min_lh: int,
) -> int:
    """폰트의 실제 한글/영문 글리프 높이를 측정해 줄 겹침·하단 잘림이 없는 안전한 줄 높이를 반환한다."""
    try:
        bb = font.getbbox("가나다Agjy()")
        glyph_h = int(bb[3] - bb[1])
        return max(min_lh, glyph_h + 7)
    except Exception:
        return min_lh


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    raw = " ".join((text or "").split())
    if not raw:
        return []
    lines: list[str] = []
    cur = ""
    for ch in raw:
        cand = cur + ch
        if draw.textlength(cand, font=font) <= max_width or not cur:
            cur = cand
        else:
            if " " in cur and not cur.endswith(" "):
                sp_idx = cur.rfind(" ")
                if sp_idx >= len(cur) // 2:
                    lines.append(cur[:sp_idx].rstrip())
                    cur = cur[sp_idx + 1 :] + ch
                    continue
            lines.append(cur.rstrip())
            cur = ch.lstrip()
    if cur:
        lines.append(cur)
    return lines


def _measure_card_height(
    draw: ImageDraw.ImageDraw,
    node: KoreanDiagramNode,
    card_w: int,
    f_sub: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    f_title: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    f_body: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    *,
    sub_h: int = 28,
    title_lh: int = 28,
    body_lh: int = 24,
    min_h: int = 112,
) -> int:
    inner_w = max(80, card_w - 34)
    h = 18
    if (node.subtitle or "").strip():
        h += sub_h
    title_lines = _wrap_text(draw, node.title, f_title, inner_w)
    h += max(1, len(title_lines)) * title_lh + 10
    for d in node.details or []:
        d_lines = _wrap_text(draw, f"• {d}", f_body, inner_w)
        h += max(1, len(d_lines)) * body_lh + 4
    h += 14
    return max(min_h, h)


def _draw_node_card(
    draw: ImageDraw.ImageDraw,
    node: KoreanDiagramNode,
    x: int,
    y: int,
    w: int,
    h: int,
    f_sub: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    f_title: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    f_body: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    *,
    sub_h: int = 30,
    title_lh: int = 28,
    body_lh: int = 24,
) -> None:
    pal = _PALETTES.get(node.accent_color, _PALETTES["blue"])
    draw.rounded_rectangle(
        [x, y, x + w, y + h],
        radius=14,
        fill=pal["bg"],
        outline=pal["border"],
        width=3,
    )
    draw.rounded_rectangle([x, y, x + w, y + 10], radius=6, fill=pal["hdr"])
    draw.rectangle([x, y + 6, x + w, y + 12], fill=pal["hdr"])

    cur_y = y + 18
    inner_x = x + 17
    inner_w = max(80, w - 34)

    sub = (node.subtitle or "").strip()
    if sub:
        badge_w = int(min(inner_w, draw.textlength(sub, font=f_sub) + 18))
        badge_h = max(23, sub_h - 4)
        draw.rounded_rectangle(
            [inner_x, cur_y, inner_x + badge_w, cur_y + badge_h],
            radius=8,
            fill=pal["badge_bg"],
        )
        draw.text((inner_x + 9, cur_y + 3), sub, font=f_sub, fill=pal["badge_fg"])
        cur_y += sub_h + 2

    for t_ln in _wrap_text(draw, node.title, f_title, inner_w):
        draw.text((inner_x, cur_y), t_ln, font=f_title, fill=(15, 23, 42))
        cur_y += title_lh
    cur_y += 5

    for d in node.details or []:
        for d_ln in _wrap_text(draw, f"• {d}", f_body, inner_w):
            draw.text((inner_x, cur_y), d_ln, font=f_body, fill=(51, 65, 85))
            cur_y += body_lh
        cur_y += 3


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    label: str = "",
    f_lbl: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None,
    color: tuple[int, int, int] = (71, 85, 105),
) -> None:
    draw.line([(x1, y1), (x2, y2)], fill=color, width=3)
    angle = math.atan2(y2 - y1, x2 - x1)
    head_len = 13
    a1 = angle + math.pi * 0.82
    a2 = angle - math.pi * 0.82
    p1 = (x2 + head_len * math.cos(a1), y2 + head_len * math.sin(a1))
    p2 = (x2 + head_len * math.cos(a2), y2 + head_len * math.sin(a2))
    draw.polygon([(x2, y2), p1, p2], fill=color)

    clean_lbl = " ".join((label or "").split()).strip()
    if clean_lbl and f_lbl is not None:
        max_pill_w = 104
        while len(clean_lbl) > 2 and draw.textlength(clean_lbl, font=f_lbl) > max_pill_w:
            clean_lbl = clean_lbl[:-1].rstrip()
        mx = (x1 + x2) // 2
        my = (y1 + y2) // 2
        tw = int(draw.textlength(clean_lbl, font=f_lbl)) + 14
        th = 22
        draw.rounded_rectangle(
            [mx - tw // 2, my - th // 2 - 18, mx + tw // 2, my + th // 2 - 18],
            radius=8,
            fill=(255, 255, 255),
            outline=(148, 163, 184),
            width=2,
        )
        draw.text((mx - tw // 2 + 7, my - th // 2 - 15), clean_lbl, font=f_lbl, fill=(30, 58, 138))
