from __future__ import annotations

import io
import os
from functools import lru_cache
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from src.models import KoreanDiagramNode, KoreanVisualDiagram

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

_REGULAR_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/NanumGothic.ttf",
)

_BOLD_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "C:/Windows/Fonts/malgunbd.ttf",
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
            # 가능하면 직전 공백에서 끊기
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
) -> int:
    inner_w = max(80, card_w - 36)
    h = 18
    if (node.subtitle or "").strip():
        h += 28
    title_lines = _wrap_text(draw, node.title, f_title, inner_w)
    h += max(1, len(title_lines)) * 30 + 10
    for d in node.details or []:
        d_lines = _wrap_text(draw, f"• {d}", f_body, inner_w)
        h += max(1, len(d_lines)) * 26 + 4
    h += 14
    return max(120, h)


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
) -> None:
    pal = _PALETTES.get(node.accent_color, _PALETTES["blue"])
    draw.rounded_rectangle(
        [x, y, x + w, y + h],
        radius=14,
        fill=pal["bg"],
        outline=pal["border"],
        width=3,
    )
    # 상단 강조 라인
    draw.rounded_rectangle(
        [x, y, x + w, y + 10],
        radius=6,
        fill=pal["hdr"],
    )
    draw.rectangle([x, y + 6, x + w, y + 12], fill=pal["hdr"])

    cur_y = y + 20
    inner_x = x + 18
    inner_w = max(80, w - 36)

    sub = (node.subtitle or "").strip()
    if sub:
        badge_w = int(min(inner_w, draw.textlength(sub, font=f_sub) + 20))
        draw.rounded_rectangle(
            [inner_x, cur_y, inner_x + badge_w, cur_y + 24],
            radius=8,
            fill=pal["badge_bg"],
        )
        draw.text((inner_x + 10, cur_y + 3), sub, font=f_sub, fill=pal["badge_fg"])
        cur_y += 32

    for t_ln in _wrap_text(draw, node.title, f_title, inner_w):
        draw.text((inner_x, cur_y), t_ln, font=f_title, fill=(15, 23, 42))
        cur_y += 30
    cur_y += 6

    for d in node.details or []:
        for d_ln in _wrap_text(draw, f"• {d}", f_body, inner_w):
            draw.text((inner_x, cur_y), d_ln, font=f_body, fill=(51, 65, 85))
            cur_y += 26
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
    import math

    angle = math.atan2(y2 - y1, x2 - x1)
    head_len = 13
    a1 = angle + math.pi * 0.82
    a2 = angle - math.pi * 0.82
    p1 = (x2 + head_len * math.cos(a1), y2 + head_len * math.sin(a1))
    p2 = (x2 + head_len * math.cos(a2), y2 + head_len * math.sin(a2))
    draw.polygon([(x2, y2), p1, p2], fill=color)

    clean_lbl = " ".join((label or "").split()).strip()
    if clean_lbl and f_lbl is not None:
        max_pill_w = 82
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


def render_korean_diagram_png(diagram: KoreanVisualDiagram) -> bytes:
    """KoreanVisualDiagram 스펙을 고해상도(1400px 폭) 한글 도식 PNG 바이트로 렌더링한다."""
    nodes = list(diagram.nodes or [])[:5]
    if not nodes:
        raise ValueError("도식에 표시할 노드가 없습니다.")

    W = 1400
    PAD_X = 36
    GAP_X = 88
    GAP_Y = 48

    f_hdr_tag = _load_font(17, bold=True)
    f_hdr_title = _load_font(26, bold=True)
    f_sub = _load_font(16, bold=True)
    f_title = _load_font(22, bold=True)
    f_body = _load_font(19, bold=False)
    f_lbl = _load_font(14, bold=True)
    f_take = _load_font(19, bold=True)

    probe = Image.new("RGB", (W, 200), (255, 255, 255))
    pdraw = ImageDraw.Draw(probe)

    title_lines = _wrap_text(pdraw, diagram.title or "핵심 원리 도식", f_hdr_title, W - PAD_X * 2 - 20)
    header_h = 26 + 26 + max(1, len(title_lines)) * 34 + 16

    takeaway_lines = (
        _wrap_text(pdraw, f"핵심 정리: {diagram.takeaway}", f_take, W - PAD_X * 2 - 44)
        if (diagram.takeaway or "").strip()
        else []
    )
    takeaway_h = (len(takeaway_lines) * 28 + 28) if takeaway_lines else 0

    dtype = diagram.diagram_type or "flow"
    n = len(nodes)
    rel_labels = list(diagram.relation_labels or [])

    # 각 레이아웃별 좌표 계산
    boxes: list[tuple[KoreanDiagramNode, int, int, int, int]] = []
    body_top = header_h + 18
    body_h = 0

    if dtype == "hierarchy" and n >= 2:
        root_w = min(680, W - PAD_X * 2)
        root_x = (W - root_w) // 2
        root_h = _measure_card_height(pdraw, nodes[0], root_w, f_sub, f_title, f_body)
        boxes.append((nodes[0], root_x, body_top, root_w, root_h))

        children = nodes[1:]
        cn = len(children)
        child_w = (W - PAD_X * 2 - GAP_X * (cn - 1)) // cn
        child_h = max(_measure_card_height(pdraw, c, child_w, f_sub, f_title, f_body) for c in children)
        child_y = body_top + root_h + 56
        for i, c_node in enumerate(children):
            cx = PAD_X + i * (child_w + GAP_X)
            boxes.append((c_node, cx, child_y, child_w, child_h))
        body_h = root_h + 56 + child_h

    elif dtype == "cycle" and n >= 3:
        # 상단 2개, 하단 1~2개 순환 배치
        cols = 2
        card_w = (W - PAD_X * 2 - 100) // cols
        top_nodes = nodes[:2]
        bot_nodes = nodes[2:4]
        top_h = max(_measure_card_height(pdraw, nd, card_w, f_sub, f_title, f_body) for nd in top_nodes)
        bot_h = max(_measure_card_height(pdraw, nd, card_w, f_sub, f_title, f_body) for nd in bot_nodes)
        boxes.append((top_nodes[0], PAD_X, body_top, card_w, top_h))
        boxes.append((top_nodes[1], W - PAD_X - card_w, body_top, card_w, top_h))
        bot_y = body_top + top_h + 72
        if len(bot_nodes) == 1:
            bx = (W - card_w) // 2
            boxes.append((bot_nodes[0], bx, bot_y, card_w, bot_h))
        else:
            boxes.append((bot_nodes[1], PAD_X, bot_y, card_w, bot_h))
            boxes.append((bot_nodes[0], W - PAD_X - card_w, bot_y, card_w, bot_h))
        body_h = top_h + 72 + bot_h

    else:
        # flow 또는 comparison (4개 이하 1줄, 5개면 상단 3개 + 하단 2개)
        if n <= 4:
            gap = 74 if dtype == "comparison" else GAP_X
            card_w = (W - PAD_X * 2 - gap * (n - 1)) // n
            card_h = max(_measure_card_height(pdraw, nd, card_w, f_sub, f_title, f_body) for nd in nodes)
            for i, nd in enumerate(nodes):
                cx = PAD_X + i * (card_w + gap)
                boxes.append((nd, cx, body_top, card_w, card_h))
            body_h = card_h
        else:
            top_nodes = nodes[:3]
            bot_nodes = nodes[3:]
            card_w = (W - PAD_X * 2 - GAP_X * 2) // 3
            top_h = max(_measure_card_height(pdraw, nd, card_w, f_sub, f_title, f_body) for nd in top_nodes)
            bot_h = max(_measure_card_height(pdraw, nd, card_w, f_sub, f_title, f_body) for nd in bot_nodes)
            for i, nd in enumerate(top_nodes):
                boxes.append((nd, PAD_X + i * (card_w + GAP_X), body_top, card_w, top_h))
            bot_y = body_top + top_h + GAP_Y + 12
            bot_start_x = (W - (len(bot_nodes) * card_w + (len(bot_nodes) - 1) * GAP_X)) // 2
            for i, nd in enumerate(bot_nodes):
                boxes.append((nd, bot_start_x + i * (card_w + GAP_X), bot_y, card_w, bot_h))
            body_h = top_h + GAP_Y + 12 + bot_h

    total_h = body_top + body_h + 24 + (takeaway_h + 18 if takeaway_h else 0)
    img = Image.new("RGB", (W, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # 외곽 프레임 & 상단 제목 헤더
    draw.rounded_rectangle(
        [8, 8, W - 9, total_h - 9],
        radius=18,
        fill=(255, 255, 255),
        outline=(203, 213, 225),
        width=3,
    )
    type_tag_map = {
        "flow": "원리 · 과정 흐름도",
        "comparison": "개념 · 이론 비교 도식",
        "cycle": "상호작용 · 순환 구조도",
        "hierarchy": "개념 체계 · 분류 구조도",
    }
    tag_text = type_tag_map.get(dtype, "개념 이해 도식")
    draw.text((PAD_X, 24), f"◆ {tag_text}", font=f_hdr_tag, fill=(29, 78, 216))
    ty = 52
    for tl in title_lines:
        draw.text((PAD_X, ty), tl, font=f_hdr_title, fill=(15, 23, 42))
        ty += 34
    draw.line([(PAD_X, header_h - 4), (W - PAD_X, header_h - 4)], fill=(226, 232, 240), width=2)

    # 카드 그리기
    for nd, bx, by, bw, bh in boxes:
        _draw_node_card(draw, nd, bx, by, bw, bh, f_sub, f_title, f_body)

    # 연결선 / 화살표 그리기
    if dtype == "hierarchy" and len(boxes) >= 2:
        _, rx, ry, rw, rh = boxes[0]
        root_cx = rx + rw // 2
        root_by = ry + rh
        mid_y = root_by + 28
        draw.line([(root_cx, root_by), (root_cx, mid_y)], fill=(71, 85, 105), width=3)
        child_centers = [(cx + cw // 2, cy) for _, cx, cy, cw, _ in boxes[1:]]
        if len(child_centers) >= 2:
            draw.line(
                [(child_centers[0][0], mid_y), (child_centers[-1][0], mid_y)],
                fill=(71, 85, 105),
                width=3,
            )
        for idx_c, (ccx, ccy) in enumerate(child_centers):
            lbl = rel_labels[idx_c] if idx_c < len(rel_labels) else ""
            _draw_arrow(draw, ccx, mid_y, ccx, ccy - 4, label=lbl, f_lbl=f_lbl)

    elif dtype == "cycle" and len(boxes) >= 3:
        for i in range(len(boxes)):
            _, x1, y1, w1, h1 = boxes[i]
            _, x2, y2, w2, h2 = boxes[(i + 1) % len(boxes)]
            lbl = rel_labels[i] if i < len(rel_labels) else ""
            c1x, c1y = x1 + w1 // 2, y1 + h1 // 2
            c2x, c2y = x2 + w2 // 2, y2 + h2 // 2
            if abs(c1y - c2y) < 40:
                if c1x < c2x:
                    _draw_arrow(draw, x1 + w1 + 4, c1y, x2 - 4, c2y, label=lbl, f_lbl=f_lbl)
                else:
                    _draw_arrow(draw, x1 - 4, c1y, x2 + w2 + 4, c2y, label=lbl, f_lbl=f_lbl)
            else:
                if c1y < c2y:
                    _draw_arrow(draw, c1x, y1 + h1 + 4, c2x, y2 - 4, label=lbl, f_lbl=f_lbl)
                else:
                    _draw_arrow(draw, c1x, y1 - 4, c2x, y2 + h2 + 4, label=lbl, f_lbl=f_lbl)

    elif dtype == "comparison":
        for i in range(len(boxes) - 1):
            _, x1, y1, w1, h1 = boxes[i]
            _, x2, y2, _, _ = boxes[i + 1]
            mx = (x1 + w1 + x2) // 2
            my = y1 + h1 // 2
            vs_lbl = " ".join((rel_labels[i] if i < len(rel_labels) else "VS").split()).strip() or "VS"
            while len(vs_lbl) > 2 and draw.textlength(vs_lbl, font=f_lbl) > 58:
                vs_lbl = vs_lbl[:-1].rstrip()
            tw = max(36, int(draw.textlength(vs_lbl, font=f_lbl)) + 16)
            draw.rounded_rectangle(
                [mx - tw // 2, my - 15, mx + tw // 2, my + 15],
                radius=11,
                fill=(239, 246, 255),
                outline=(59, 130, 246),
                width=2,
            )
            draw.text((mx - tw // 2 + 8, my - 8), vs_lbl, font=f_lbl, fill=(29, 78, 216))

    else:
        # flow 화살표
        for i in range(len(boxes) - 1):
            _, x1, y1, w1, h1 = boxes[i]
            _, x2, y2, w2, h2 = boxes[i + 1]
            lbl = rel_labels[i] if i < len(rel_labels) else ""
            if abs(y1 - y2) < 40:
                _draw_arrow(draw, x1 + w1 + 4, y1 + h1 // 2, x2 - 4, y2 + h2 // 2, label=lbl, f_lbl=f_lbl)
            else:
                _draw_arrow(draw, x1 + w1 // 2, y1 + h1 + 4, x2 + w2 // 2, y2 - 4, label=lbl, f_lbl=f_lbl)

    # 하단 핵심 정리 배너
    if takeaway_lines:
        ty_box = body_top + body_h + 22
        draw.rounded_rectangle(
            [PAD_X, ty_box, W - PAD_X, ty_box + takeaway_h],
            radius=12,
            fill=(255, 251, 235),
            outline=(245, 158, 11),
            width=2,
        )
        cur_ty = ty_box + 14
        for tl in takeaway_lines:
            draw.text((PAD_X + 20, cur_ty), tl, font=f_take, fill=(146, 64, 14))
            cur_ty += 28

    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()
