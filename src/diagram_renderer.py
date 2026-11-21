from __future__ import annotations

import io
import math
import os
from functools import lru_cache
from typing import Any

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

_COLOR_CYCLE = ("blue", "green", "amber", "purple", "slate")

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
    inner_w = max(80, card_w - 34)
    h = 18
    if (node.subtitle or "").strip():
        h += 28
    title_lines = _wrap_text(draw, node.title, f_title, inner_w)
    h += max(1, len(title_lines)) * 28 + 10
    for d in node.details or []:
        d_lines = _wrap_text(draw, f"• {d}", f_body, inner_w)
        h += max(1, len(d_lines)) * 24 + 4
    h += 14
    return max(112, h)


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
    draw.rounded_rectangle([x, y, x + w, y + 10], radius=6, fill=pal["hdr"])
    draw.rectangle([x, y + 6, x + w, y + 12], fill=pal["hdr"])

    cur_y = y + 18
    inner_x = x + 17
    inner_w = max(80, w - 34)

    sub = (node.subtitle or "").strip()
    if sub:
        badge_w = int(min(inner_w, draw.textlength(sub, font=f_sub) + 18))
        draw.rounded_rectangle(
            [inner_x, cur_y, inner_x + badge_w, cur_y + 23],
            radius=8,
            fill=pal["badge_bg"],
        )
        draw.text((inner_x + 9, cur_y + 3), sub, font=f_sub, fill=pal["badge_fg"])
        cur_y += 30

    for t_ln in _wrap_text(draw, node.title, f_title, inner_w):
        draw.text((inner_x, cur_y), t_ln, font=f_title, fill=(15, 23, 42))
        cur_y += 28
    cur_y += 5

    for d in node.details or []:
        for d_ln in _wrap_text(draw, f"• {d}", f_body, inner_w):
            draw.text((inner_x, cur_y), d_ln, font=f_body, fill=(51, 65, 85))
            cur_y += 24
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


def _render_matrix(
    diagram: KoreanVisualDiagram,
    W: int,
    PAD_X: int,
    body_top: int,
    fonts: dict[str, Any],
) -> tuple[int, list[Any]]:
    """항목별 정밀 비교 매트릭스표 레이아웃을 계산한다."""
    nodes = list(diagram.nodes or [])[:4]
    col_cnt = max(2, len(nodes))
    while len(nodes) < col_cnt:
        nodes.append(KoreanDiagramNode(title=f"대상 {len(nodes)+1}"))

    rows = list(diagram.matrix_rows or [])
    if not rows:
        max_d = max((len(nd.details or []) for nd in nodes), default=1)
        for ri in range(max(1, min(4, max_d))):
            rows.append(
                type("_Row", (), {
                    "criterion": f"핵심 {ri + 1}",
                    "values": [
                        (nd.details[ri] if ri < len(nd.details or []) else "-")
                        for nd in nodes
                    ],
                })()
            )

    probe = ImageDraw.Draw(Image.new("RGB", (W, 100)))
    table_w = W - PAD_X * 2
    crit_w = 210
    val_w = (table_w - crit_w) // col_cnt
    f_hdr = fonts["title"]
    f_body = fonts["body"]
    f_sub = fonts["sub"]

    hdr_h = 56
    row_heights: list[int] = []
    for r in rows[:6]:
        c_lines = _wrap_text(probe, r.criterion, f_sub, crit_w - 24)
        max_lines = len(c_lines)
        for ci in range(col_cnt):
            val_txt = r.values[ci] if ci < len(r.values) else "-"
            v_lines = _wrap_text(probe, val_txt, f_body, val_w - 26)
            max_lines = max(max_lines, len(v_lines))
        row_heights.append(max(50, max_lines * 25 + 22))

    total_h = hdr_h + sum(row_heights)

    def _draw_fn(draw: ImageDraw.ImageDraw) -> None:
        x0 = PAD_X
        y0 = body_top
        # 헤더 행
        draw.rounded_rectangle(
            [x0, y0, x0 + crit_w - 2, y0 + hdr_h],
            radius=8,
            fill=(30, 41, 59),
        )
        draw.text((x0 + 18, y0 + 16), "비교 기준", font=f_hdr, fill=(255, 255, 255))

        for ci, nd in enumerate(nodes):
            pal = _PALETTES.get(nd.accent_color or _COLOR_CYCLE[ci % len(_COLOR_CYCLE)], _PALETTES["blue"])
            cx = x0 + crit_w + ci * val_w + 2
            draw.rounded_rectangle(
                [cx, y0, cx + val_w - 4, y0 + hdr_h],
                radius=8,
                fill=pal["hdr"],
            )
            t_lines = _wrap_text(draw, nd.title, f_hdr, val_w - 24)
            draw.text((cx + 14, y0 + (16 if len(t_lines) == 1 else 6)), t_lines[0], font=f_hdr, fill=(255, 255, 255))

        cur_y = y0 + hdr_h + 4
        for ri, r in enumerate(rows[:6]):
            rh = row_heights[ri]
            bg_c = (248, 250, 252) if ri % 2 == 0 else (255, 255, 255)
            draw.rectangle(
                [x0, cur_y, x0 + table_w, cur_y + rh],
                fill=bg_c,
                outline=(203, 213, 225),
                width=2,
            )
            draw.rectangle([x0, cur_y, x0 + crit_w, cur_y + rh], fill=(241, 245, 249), outline=(203, 213, 225), width=2)
            cy_txt = cur_y + 12
            for cl in _wrap_text(draw, r.criterion, f_sub, crit_w - 24):
                draw.text((x0 + 14, cy_txt), cl, font=f_sub, fill=(30, 58, 138))
                cy_txt += 24

            for ci in range(col_cnt):
                cx = x0 + crit_w + ci * val_w
                draw.line([(cx, cur_y), (cx, cur_y + rh)], fill=(203, 213, 225), width=2)
                val_txt = r.values[ci] if ci < len(r.values) else "-"
                vy = cur_y + 12
                for vl in _wrap_text(draw, val_txt, f_body, val_w - 26):
                    draw.text((cx + 14, vy), vl, font=f_body, fill=(30, 41, 59))
                    vy += 24
            cur_y += rh

    return total_h, [_draw_fn]


def _render_graph(
    diagram: KoreanVisualDiagram,
    W: int,
    PAD_X: int,
    body_top: int,
    fonts: dict[str, Any],
) -> tuple[int, list[Any]]:
    """X-Y 좌표 함수·비례 그래프 + 우측 핵심 해석 카드 레이아웃."""
    gspec = diagram.graph_spec
    x_lbl = gspec.x_label if gspec else "독립 변수 (X)"
    y_lbl = gspec.y_label if gspec else "종속 변수 (Y)"
    curves = list(gspec.curves or []) if gspec else []
    if not curves:
        from src.models import KoreanGraphCurve
        curves = [KoreanGraphCurve(label=diagram.title or "핵심 관계", trend="up", accent_color="blue")]
    hi_pt = (gspec.highlight_point if gspec else "") or ""

    probe = ImageDraw.Draw(Image.new("RGB", (W, 100)))
    plot_w = 720
    plot_h = 370
    side_x = PAD_X + plot_w + 32
    side_w = W - PAD_X - side_x

    nodes = list(diagram.nodes or [])[:3]
    if not nodes:
        nodes = [
            KoreanDiagramNode(
                title=c.label,
                subtitle="변화 추세",
                details=[c.annotation] if c.annotation else [],
                accent_color=c.accent_color,
            )
            for c in curves[:2]
        ]
    if hi_pt:
        nodes.append(KoreanDiagramNode(title="핵심 균형·임계점", subtitle="출제 포인트", details=[hi_pt], accent_color="amber"))

    card_heights = [
        _measure_card_height(probe, nd, side_w, fonts["sub"], fonts["title"], fonts["body"])
        for nd in nodes
    ]
    side_total_h = sum(card_heights) + max(0, len(nodes) - 1) * 14
    total_h = max(plot_h + 30, side_total_h)

    def _draw_fn(draw: ImageDraw.ImageDraw) -> None:
        gx0 = PAD_X + 64
        gy0 = body_top + 28
        gx1 = PAD_X + plot_w - 28
        gy1 = body_top + plot_h - 46

        # 배경 & 격자
        draw.rounded_rectangle(
            [PAD_X, body_top, PAD_X + plot_w, body_top + plot_h],
            radius=14,
            fill=(248, 250, 252),
            outline=(203, 213, 225),
            width=2,
        )
        for frac in (0.25, 0.5, 0.75):
            lx = int(gx0 + (gx1 - gx0) * frac)
            ly = int(gy0 + (gy1 - gy0) * frac)
            draw.line([(lx, gy0), (lx, gy1)], fill=(226, 232, 240), width=1)
            draw.line([(gx0, ly), (gx1, ly)], fill=(226, 232, 240), width=1)

        # X축, Y축 화살표
        _draw_arrow(draw, gx0, gy1, gx0, gy0 - 12, color=(30, 41, 59))
        _draw_arrow(draw, gx0, gy1, gx1 + 14, gy1, color=(30, 41, 59))
        draw.text((gx0 - 44, gy0 - 22), y_lbl[:14], font=fonts["sub"], fill=(15, 23, 42))
        x_tw = draw.textlength(x_lbl[:18], font=fonts["sub"])
        draw.text((gx1 - x_tw + 8, gy1 + 12), x_lbl[:18], font=fonts["sub"], fill=(15, 23, 42))

        # 곡선 그리기
        steps = 40
        for ci, c in enumerate(curves[:2]):
            pal = _PALETTES.get(c.accent_color, _PALETTES["blue"])
            pts: list[tuple[float, float]] = []
            for s in range(steps + 1):
                t = s / steps
                px = gx0 + 18 + t * (gx1 - gx0 - 36)
                if c.trend == "down":
                    norm_y = 0.85 - 0.68 * (t ** 0.85)
                elif c.trend == "u_shape":
                    norm_y = 0.22 + 2.4 * ((t - 0.5) ** 2)
                elif c.trend == "inv_u":
                    norm_y = 0.85 - 2.4 * ((t - 0.5) ** 2)
                elif c.trend == "flat":
                    norm_y = 0.52 if ci == 0 else 0.35
                else:
                    norm_y = 0.16 + 0.68 * (t ** 0.9)
                if ci == 1 and c.trend in ("up", "down"):
                    norm_y = max(0.1, min(0.9, norm_y + 0.08))
                py = gy1 - norm_y * (gy1 - gy0 - 20)
                pts.append((px, py))
            draw.line(pts, fill=pal["hdr"], width=5)
            # 곡선 끝 라벨
            lx, ly = pts[-1]
            draw.text((min(gx1 - 120, lx - 80), max(gy0, ly - 24)), c.label[:14], font=fonts["sub"], fill=pal["hdr"])

        if hi_pt:
            mx, my = (gx0 + gx1) // 2, (gy0 + gy1) // 2
            draw.ellipse([mx - 8, my - 8, mx + 8, my + 8], fill=(245, 158, 11), outline=(146, 64, 14), width=3)

        # 우측 설명 카드
        cy = body_top
        for idx_n, nd in enumerate(nodes):
            ch = card_heights[idx_n]
            _draw_node_card(draw, nd, side_x, cy, side_w, ch, fonts["sub"], fonts["title"], fonts["body"])
            cy += ch + 14

    return total_h, [_draw_fn]


def _render_anatomy(
    diagram: KoreanVisualDiagram,
    W: int,
    PAD_X: int,
    body_top: int,
    fonts: dict[str, Any],
) -> tuple[int, list[Any]]:
    """중앙 핵심 장치/원리 코어 + 주변 지시선(Call-out) 구조 모식도."""
    nodes = list(diagram.nodes or [])[:4]
    center_txt = (diagram.center_label or diagram.title or "핵심 구조").strip()
    probe = ImageDraw.Draw(Image.new("RGB", (W, 100)))

    card_w = 430
    core_w = 340
    core_h = 150

    left_nodes = nodes[::2]
    right_nodes = nodes[1::2]
    left_hs = [_measure_card_height(probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"]) for nd in left_nodes]
    right_hs = [_measure_card_height(probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"]) for nd in right_nodes]

    left_total = sum(left_hs) + max(0, len(left_hs) - 1) * 26
    right_total = sum(right_hs) + max(0, len(right_hs) - 1) * 26
    total_h = max(core_h + 60, left_total, right_total)

    def _draw_fn(draw: ImageDraw.ImageDraw) -> None:
        core_x = (W - core_w) // 2
        core_y = body_top + (total_h - core_h) // 2
        core_cx = core_x + core_w // 2
        core_cy = core_y + core_h // 2

        # 좌측 카드 + 지시선
        ly = body_top + (total_h - left_total) // 2
        for idx_l, nd in enumerate(left_nodes):
            ch = left_hs[idx_l]
            lx = PAD_X
            _draw_node_card(draw, nd, lx, ly, card_w, ch, fonts["sub"], fonts["title"], fonts["body"])
            sx, sy = lx + card_w, ly + ch // 2
            tx, ty = core_x - 6, core_cy + (-22 if idx_l == 0 and len(left_nodes) > 1 else (22 if idx_l > 0 else 0))
            draw.line([(sx, sy), (tx, ty)], fill=(59, 130, 246), width=3)
            draw.ellipse([tx - 6, ty - 6, tx + 6, ty + 6], fill=(29, 78, 216))
            ly += ch + 26

        # 우측 카드 + 지시선
        ry = body_top + (total_h - right_total) // 2
        for idx_r, nd in enumerate(right_nodes):
            ch = right_hs[idx_r]
            rx = W - PAD_X - card_w
            _draw_node_card(draw, nd, rx, ry, card_w, ch, fonts["sub"], fonts["title"], fonts["body"])
            sx, sy = rx, ry + ch // 2
            tx, ty = core_x + core_w + 6, core_cy + (-22 if idx_r == 0 and len(right_nodes) > 1 else (22 if idx_r > 0 else 0))
            draw.line([(sx, sy), (tx, ty)], fill=(34, 197, 94), width=3)
            draw.ellipse([tx - 6, ty - 6, tx + 6, ty + 6], fill=(21, 128, 61))
            ry += ch + 26

        # 중앙 코어 박스 (이중 원형/둥근 모형)
        draw.rounded_rectangle(
            [core_x, core_y, core_x + core_w, core_y + core_h],
            radius=28,
            fill=(239, 246, 255),
            outline=(29, 78, 216),
            width=4,
        )
        draw.rounded_rectangle(
            [core_x + 8, core_y + 8, core_x + core_w - 8, core_y + core_h - 8],
            radius=22,
            fill=(255, 255, 255),
            outline=(147, 197, 253),
            width=2,
        )
        c_lines = _wrap_text(draw, center_txt, fonts["title"], core_w - 44)
        cy_txt = core_cy - (len(c_lines) * 28) // 2
        for cl in c_lines:
            tw = draw.textlength(cl, font=fonts["title"])
            draw.text((core_cx - tw / 2, cy_txt), cl, font=fonts["title"], fill=(30, 58, 138))
            cy_txt += 28

    return total_h, [_draw_fn]


def _render_timeline(
    diagram: KoreanVisualDiagram,
    W: int,
    PAD_X: int,
    body_top: int,
    fonts: dict[str, Any],
) -> tuple[int, list[Any]]:
    """시대·단계별 연표 타임라인 레이아웃."""
    nodes = list(diagram.nodes or [])[:4]
    n = max(1, len(nodes))
    probe = ImageDraw.Draw(Image.new("RGB", (W, 100)))
    gap = 28
    card_w = (W - PAD_X * 2 - gap * (n - 1)) // n
    card_h = max(_measure_card_height(probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"]) for nd in nodes)
    axis_h = 54
    total_h = axis_h + card_h

    def _draw_fn(draw: ImageDraw.ImageDraw) -> None:
        axis_y = body_top + 24
        draw.line([(PAD_X + 40, axis_y), (W - PAD_X - 40, axis_y)], fill=(59, 130, 246), width=5)
        card_y = body_top + axis_h
        for i, nd in enumerate(nodes):
            cx = PAD_X + i * (card_w + gap)
            mid_x = cx + card_w // 2
            pal = _PALETTES.get(nd.accent_color, _PALETTES["blue"])
            draw.line([(mid_x, axis_y), (mid_x, card_y)], fill=pal["border"], width=3)
            draw.ellipse([mid_x - 16, axis_y - 16, mid_x + 16, axis_y + 16], fill=pal["hdr"], outline=(255, 255, 255), width=3)
            num_s = str(i + 1)
            nw = draw.textlength(num_s, font=fonts["sub"])
            draw.text((mid_x - nw / 2, axis_y - 9), num_s, font=fonts["sub"], fill=(255, 255, 255))
            _draw_node_card(draw, nd, cx, card_y, card_w, card_h, fonts["sub"], fonts["title"], fonts["body"])

    return total_h, [_draw_fn]


def _render_venn(
    diagram: KoreanVisualDiagram,
    W: int,
    PAD_X: int,
    body_top: int,
    fonts: dict[str, Any],
) -> tuple[int, list[Any]]:
    """개념 포함·교집합 벤다이어그램 + 하단 카드 레이아웃."""
    nodes = list(diagram.nodes or [])[:3]
    probe = ImageDraw.Draw(Image.new("RGB", (W, 100)))
    venn_h = 250
    n = max(1, len(nodes))
    gap = 32
    card_w = (W - PAD_X * 2 - gap * (n - 1)) // n
    card_h = max(_measure_card_height(probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"]) for nd in nodes)
    total_h = venn_h + 18 + card_h

    def _draw_fn(draw: ImageDraw.ImageDraw) -> None:
        # 2원 교집합 기본 시각화 (3개일 때도 핵심 2원+교집합/3원 겹침 표현)
        r = 115
        cy = body_top + 125
        if len(nodes) >= 2:
            c1x = W // 2 - 78
            c2x = W // 2 + 78
            # 연한 채우기 + 테두리
            draw.ellipse([c1x - r, cy - r, c1x + r, cy + r], fill=(219, 234, 254), outline=(37, 99, 235), width=4)
            draw.ellipse([c2x - r, cy - r, c2x + r, cy + r], fill=(220, 252, 231), outline=(22, 163, 74), width=4)
            # 교집합 겹침 영역 강조선
            draw.ellipse([c1x - r, cy - r, c1x + r, cy + r], outline=(37, 99, 235), width=4)
            t1 = nodes[0].title[:12]
            t2 = nodes[1].title[:12]
            draw.text((c1x - 90, cy - 12), t1, font=fonts["title"], fill=(30, 58, 138))
            draw.text((c2x + 16, cy - 12), t2, font=fonts["title"], fill=(22, 101, 52))
            center_lbl = (diagram.center_label or "공통 핵심").strip()[:12]
            cw = draw.textlength(center_lbl, font=fonts["sub"])
            draw.rounded_rectangle(
                [W // 2 - cw / 2 - 12, cy - 16, W // 2 + cw / 2 + 12, cy + 16],
                radius=10,
                fill=(255, 251, 235),
                outline=(217, 119, 6),
                width=2,
            )
            draw.text((W // 2 - cw / 2, cy - 9), center_lbl, font=fonts["sub"], fill=(146, 64, 14))

        card_y = body_top + venn_h + 18
        for i, nd in enumerate(nodes):
            _draw_node_card(draw, nd, PAD_X + i * (card_w + gap), card_y, card_w, card_h, fonts["sub"], fonts["title"], fonts["body"])

    return total_h, [_draw_fn]


def _render_scale(
    diagram: KoreanVisualDiagram,
    W: int,
    PAD_X: int,
    body_top: int,
    fonts: dict[str, Any],
) -> tuple[int, list[Any]]:
    """상충 가치·트레이드오프 대립 균형 저울도."""
    nodes = list(diagram.nodes or [])[:2]
    while len(nodes) < 2:
        nodes.append(KoreanDiagramNode(title="대립 입장"))
    probe = ImageDraw.Draw(Image.new("RGB", (W, 100)))
    card_w = 540
    card_h = max(_measure_card_height(probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"]) for nd in nodes)
    tilt = diagram.scale_tilt or "balanced"
    l_dy = 22 if tilt == "left" else (-12 if tilt == "right" else 0)
    r_dy = 22 if tilt == "right" else (-12 if tilt == "left" else 0)
    beam_area_h = 90
    total_h = card_h + beam_area_h + 24

    def _draw_fn(draw: ImageDraw.ImageDraw) -> None:
        lx = PAD_X
        rx = W - PAD_X - card_w
        ly = body_top + 12 + l_dy
        ry = body_top + 12 + r_dy
        _draw_node_card(draw, nodes[0], lx, ly, card_w, card_h, fonts["sub"], fonts["title"], fonts["body"])
        _draw_node_card(draw, nodes[1], rx, ry, card_w, card_h, fonts["sub"], fonts["title"], fonts["body"])

        l_mid = lx + card_w // 2
        r_mid = rx + card_w // 2
        beam_ly = ly + card_h + 18
        beam_ry = ry + card_h + 18
        draw.line([(l_mid, ly + card_h), (l_mid, beam_ly)], fill=(100, 116, 139), width=4)
        draw.line([(r_mid, ry + card_h), (r_mid, beam_ry)], fill=(100, 116, 139), width=4)
        draw.line([(l_mid, beam_ly), (r_mid, beam_ry)], fill=(30, 41, 59), width=6)

        pivot_x = W // 2
        pivot_y = (beam_ly + beam_ry) // 2
        base_y = body_top + total_h - 6
        draw.polygon([(pivot_x, pivot_y), (pivot_x - 36, base_y), (pivot_x + 36, base_y)], fill=(71, 85, 105))
        draw.ellipse([pivot_x - 10, pivot_y - 10, pivot_x + 10, pivot_y + 10], fill=(245, 158, 11), outline=(30, 41, 59), width=3)

    return total_h, [_draw_fn]


def _render_fishbone(
    diagram: KoreanVisualDiagram,
    W: int,
    PAD_X: int,
    body_top: int,
    fonts: dict[str, Any],
) -> tuple[int, list[Any]]:
    """피쉬본(어골도): 다발적 원인·요인 -> 핵심 결과/현상 수렴 도식."""
    nodes = list(diagram.nodes or [])[:4]
    while len(nodes) < 2:
        nodes.append(KoreanDiagramNode(title=f"핵심 요인 {len(nodes) + 1}"))
    head_txt = (diagram.center_label or diagram.title or "핵심 결과").strip()
    probe = ImageDraw.Draw(Image.new("RGB", (W, 100)))

    card_w = 410
    head_w = 270
    head_h = 136
    top_nodes = nodes[::2]
    bot_nodes = nodes[1::2]
    top_h = max((_measure_card_height(probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"]) for nd in top_nodes), default=120)
    bot_h = max((_measure_card_height(probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"]) for nd in bot_nodes), default=120)
    spine_gap = 76
    total_h = top_h + spine_gap + bot_h

    def _draw_fn(draw: ImageDraw.ImageDraw) -> None:
        spine_y = body_top + top_h + spine_gap // 2
        head_x = W - PAD_X - head_w
        head_y = spine_y - head_h // 2

        # 중앙 척추선
        _draw_arrow(draw, PAD_X + 20, spine_y, head_x - 8, spine_y, color=(30, 41, 59))

        # 상단 원인 카드 + 대각선 가시
        for i, nd in enumerate(top_nodes):
            cx = PAD_X + 30 + i * (card_w + 56)
            cy = body_top + (top_h - _measure_card_height(probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"]))
            ch = _measure_card_height(probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"])
            _draw_node_card(draw, nd, cx, cy, card_w, ch, fonts["sub"], fonts["title"], fonts["body"])
            sx = cx + card_w // 2
            tx = min(head_x - 34, sx + 70)
            draw.line([(sx, cy + ch), (tx, spine_y)], fill=(59, 130, 246), width=3)
            draw.ellipse([tx - 5, spine_y - 5, tx + 5, spine_y + 5], fill=(29, 78, 216))

        # 하단 원인 카드 + 대각선 가시
        for i, nd in enumerate(bot_nodes):
            cx = PAD_X + 30 + i * (card_w + 56)
            cy = body_top + top_h + spine_gap
            ch = _measure_card_height(probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"])
            _draw_node_card(draw, nd, cx, cy, card_w, ch, fonts["sub"], fonts["title"], fonts["body"])
            sx = cx + card_w // 2
            tx = min(head_x - 34, sx + 70)
            draw.line([(sx, cy), (tx, spine_y)], fill=(22, 163, 74), width=3)
            draw.ellipse([tx - 5, spine_y - 5, tx + 5, spine_y + 5], fill=(21, 128, 61))

        # 우측 핵심 결과(머리) 박스
        draw.rounded_rectangle(
            [head_x, head_y, head_x + head_w, head_y + head_h],
            radius=20,
            fill=(255, 251, 235),
            outline=(217, 119, 6),
            width=4,
        )
        draw.text((head_x + 18, head_y + 14), "핵심 결과 · 현상", font=fonts["sub"], fill=(180, 83, 9))
        h_lines = _wrap_text(draw, head_txt, fonts["title"], head_w - 36)
        hy = head_y + 48
        for hl in h_lines[:3]:
            draw.text((head_x + 18, hy), hl, font=fonts["title"], fill=(15, 23, 42))
            hy += 28

    return total_h, [_draw_fn]


def _render_quadrant(
    diagram: KoreanVisualDiagram,
    W: int,
    PAD_X: int,
    body_top: int,
    fonts: dict[str, Any],
) -> tuple[int, list[Any]]:
    """2x2 사분면 분류 도식: 두 개의 직교 분류 축을 기준으로 4개 구역에 개념 배치."""
    nodes = list(diagram.nodes or [])[:4]
    while len(nodes) < 4:
        nodes.append(KoreanDiagramNode(title=f"유형 {len(nodes) + 1}", accent_color=_COLOR_CYCLE[len(nodes) % len(_COLOR_CYCLE)]))

    y_lbl = (diagram.quadrant_y_label or "세로 분류 기준 (Y)").strip()
    x_lbl = (diagram.quadrant_x_label or "가로 분류 기준 (X)").strip()
    probe = ImageDraw.Draw(Image.new("RGB", (W, 100)))

    card_w = 570
    top_h = max(_measure_card_height(probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"]) for nd in nodes[:2])
    bot_h = max(_measure_card_height(probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"]) for nd in nodes[2:4])
    top_pad = 44
    mid_gap = 64
    bot_pad = 40
    total_h = top_pad + top_h + mid_gap + bot_h + bot_pad

    def _draw_fn(draw: ImageDraw.ImageDraw) -> None:
        cx = W // 2
        cy = body_top + top_pad + top_h + mid_gap // 2

        # 십자 축 화살표
        _draw_arrow(draw, cx, body_top + total_h - bot_pad + 6, cx, body_top + 28, color=(30, 41, 59))
        _draw_arrow(draw, PAD_X + 10, cy, W - PAD_X - 10, cy, color=(30, 41, 59))

        # Y축 기준 배지 (상단 중앙)
        ytw = draw.textlength(y_lbl, font=fonts["sub"])
        draw.rounded_rectangle(
            [cx - ytw / 2 - 14, body_top, cx + ytw / 2 + 14, body_top + 26],
            radius=8,
            fill=(239, 246, 255),
            outline=(59, 130, 246),
            width=2,
        )
        draw.text((cx - ytw / 2, body_top + 4), y_lbl, font=fonts["sub"], fill=(29, 78, 216))

        # X축 기준 배지 (하단 중앙)
        xtw = draw.textlength(x_lbl, font=fonts["sub"])
        xy_box = body_top + total_h - 30
        draw.rounded_rectangle(
            [cx - xtw / 2 - 14, xy_box, cx + xtw / 2 + 14, xy_box + 26],
            radius=8,
            fill=(240, 253, 244),
            outline=(34, 197, 94),
            width=2,
        )
        draw.text((cx - xtw / 2, xy_box + 4), x_lbl, font=fonts["sub"], fill=(22, 101, 52))

        # 4개 사분면 카드 (좌상, 우상, 좌하, 우하)
        lx = PAD_X + 24
        rx = W - PAD_X - 24 - card_w
        ty = body_top + top_pad
        by = body_top + top_pad + top_h + mid_gap
        _draw_node_card(draw, nodes[0], lx, ty, card_w, top_h, fonts["sub"], fonts["title"], fonts["body"])
        _draw_node_card(draw, nodes[1], rx, ty, card_w, top_h, fonts["sub"], fonts["title"], fonts["body"])
        _draw_node_card(draw, nodes[2], lx, by, card_w, bot_h, fonts["sub"], fonts["title"], fonts["body"])
        _draw_node_card(draw, nodes[3], rx, by, card_w, bot_h, fonts["sub"], fonts["title"], fonts["body"])

    return total_h, [_draw_fn]


def _render_decision_tree(
    diagram: KoreanVisualDiagram,
    W: int,
    PAD_X: int,
    body_top: int,
    fonts: dict[str, Any],
) -> tuple[int, list[Any]]:
    """조건 분기(If-Else) 결정 트리: 상단 조건 판단 마름모 -> 예/아니오 분기 결과."""
    nodes = list(diagram.nodes or [])[:3]
    while len(nodes) < 2:
        nodes.append(KoreanDiagramNode(title=f"결과 {len(nodes) + 1}"))
    cond_txt = (diagram.center_label or diagram.title or "핵심 조건 판단").strip()
    rel_labels = list(diagram.relation_labels or [])
    probe = ImageDraw.Draw(Image.new("RGB", (W, 100)))

    diamond_h = 128
    diamond_w = 420
    branch_gap = 64
    n = len(nodes)
    card_gap = 46
    card_w = (W - PAD_X * 2 - card_gap * (n - 1)) // n
    card_h = max(_measure_card_height(probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"]) for nd in nodes)
    total_h = diamond_h + branch_gap + card_h

    def _draw_fn(draw: ImageDraw.ImageDraw) -> None:
        cx = W // 2
        dy_top = body_top
        dy_mid = dy_top + diamond_h // 2
        dy_bot = dy_top + diamond_h

        # 조건 마름모(◇)
        pts = [(cx, dy_top), (cx + diamond_w // 2, dy_mid), (cx, dy_bot), (cx - diamond_w // 2, dy_mid)]
        draw.polygon(pts, fill=(239, 246, 255), outline=(29, 78, 216), width=4)
        c_lines = _wrap_text(draw, cond_txt, fonts["title"], diamond_w - 130)
        cy_t = dy_mid - (len(c_lines[:2]) * 26) // 2
        for cl in c_lines[:2]:
            tw = draw.textlength(cl, font=fonts["title"])
            draw.text((cx - tw / 2, cy_t), cl, font=fonts["title"], fill=(15, 23, 42))
            cy_t += 26

        # 분기선 + 결과 카드
        mid_y = dy_bot + branch_gap // 2
        draw.line([(cx, dy_bot), (cx, mid_y)], fill=(71, 85, 105), width=3)
        card_y = dy_bot + branch_gap
        centers = [PAD_X + i * (card_w + card_gap) + card_w // 2 for i in range(n)]
        draw.line([(centers[0], mid_y), (centers[-1], mid_y)], fill=(71, 85, 105), width=3)

        default_branches = ["예 (O)", "아니오 (X)", "조건 분기"] if n == 2 else ["조건 1", "조건 2", "조건 3"]
        for i, nd in enumerate(nodes):
            ccx = centers[i]
            lbl = rel_labels[i] if i < len(rel_labels) and rel_labels[i].strip() else default_branches[i]
            _draw_arrow(draw, ccx, mid_y, ccx, card_y - 4, label=lbl, f_lbl=fonts["lbl"])
            _draw_node_card(draw, nd, PAD_X + i * (card_w + card_gap), card_y, card_w, card_h, fonts["sub"], fonts["title"], fonts["body"])

    return total_h, [_draw_fn]


def _render_pyramid(
    diagram: KoreanVisualDiagram,
    W: int,
    PAD_X: int,
    body_top: int,
    fonts: dict[str, Any],
) -> tuple[int, list[Any]]:
    """위계·계층 피라미드: 최상위(좁음) -> 최하위(넓음) 다층 레이어 도식."""
    nodes = list(diagram.nodes or [])[:4]
    while len(nodes) < 2:
        nodes.append(KoreanDiagramNode(title=f"계층 {len(nodes) + 1}"))
    n = len(nodes)
    probe = ImageDraw.Draw(Image.new("RGB", (W, 100)))

    layer_heights: list[int] = []
    min_w = 540
    max_w = W - PAD_X * 2
    for i, nd in enumerate(nodes):
        top_w = int(min_w + (max_w - min_w) * (i / max(1, n)))
        avail_w = max(240, top_w - 80)
        d_txt = " · ".join(nd.details or [])
        t_lines = _wrap_text(probe, nd.title, fonts["title"], avail_w)
        d_lines = _wrap_text(probe, d_txt, fonts["body"], avail_w) if d_txt else []
        lh = max(76, 22 + len(t_lines) * 28 + len(d_lines) * 24 + 14)
        layer_heights.append(lh)

    gap_y = 8
    total_h = sum(layer_heights) + gap_y * (n - 1)

    def _draw_fn(draw: ImageDraw.ImageDraw) -> None:
        cur_y = body_top
        cx = W // 2
        for i, nd in enumerate(nodes):
            lh = layer_heights[i]
            w_top = int(min_w + (max_w - min_w) * (i / n))
            w_bot = int(min_w + (max_w - min_w) * ((i + 1) / n))
            pal = _PALETTES.get(nd.accent_color or _COLOR_CYCLE[i % len(_COLOR_CYCLE)], _PALETTES["blue"])
            poly = [
                (cx - w_top // 2, cur_y),
                (cx + w_top // 2, cur_y),
                (cx + w_bot // 2, cur_y + lh),
                (cx - w_bot // 2, cur_y + lh),
            ]
            draw.polygon(poly, fill=pal["bg"], outline=pal["border"], width=3)

            avail_w = max(220, w_top - 70)
            t_lines = _wrap_text(draw, nd.title, fonts["title"], avail_w)
            d_txt = " · ".join(nd.details or [])
            d_lines = _wrap_text(draw, d_txt, fonts["body"], avail_w) if d_txt else []
            txt_block_h = len(t_lines) * 28 + (len(d_lines) * 24 + 4 if d_lines else 0)
            ty = cur_y + max(12, (lh - txt_block_h) // 2)
            for tl in t_lines:
                tw = draw.textlength(tl, font=fonts["title"])
                draw.text((cx - tw / 2, ty), tl, font=fonts["title"], fill=pal["hdr"])
                ty += 28
            if d_lines:
                ty += 4
                for dl in d_lines:
                    dw = draw.textlength(dl, font=fonts["body"])
                    draw.text((cx - dw / 2, ty), dl, font=fonts["body"], fill=(30, 41, 59))
                    ty += 24
            cur_y += lh + gap_y

    return total_h, [_draw_fn]


def _render_character_map(
    diagram: KoreanVisualDiagram,
    W: int,
    PAD_X: int,
    body_top: int,
    fonts: dict[str, Any],
) -> tuple[int, list[Any]]:
    """문학 소설 인물관계도 · 갈등 구조도: 인물 노드 + 조력(+) / 대립(-) 관계 화살표."""
    nodes = list(diagram.nodes or [])[:4]
    while len(nodes) < 2:
        nodes.append(KoreanDiagramNode(title=f"인물 {len(nodes) + 1}"))
    n = len(nodes)
    probe = ImageDraw.Draw(Image.new("RGB", (W, 100)))

    gap_x = 180
    gap_y = 96
    card_w = (W - PAD_X * 2 - gap_x) // 2

    boxes: list[tuple[int, int, int, int]] = []
    if n == 2:
        row_h = max(_measure_card_height(probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"]) for nd in nodes)
        boxes.append((PAD_X, body_top, card_w, row_h))
        boxes.append((W - PAD_X - card_w, body_top, card_w, row_h))
        body_h = row_h
    else:
        top_nodes = nodes[:2]
        bot_nodes = nodes[2:4]
        top_h = max(_measure_card_height(probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"]) for nd in top_nodes)
        bot_h = max(_measure_card_height(probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"]) for nd in bot_nodes)
        boxes.append((PAD_X, body_top, card_w, top_h))
        boxes.append((W - PAD_X - card_w, body_top, card_w, top_h))
        bot_y = body_top + top_h + gap_y
        if len(bot_nodes) == 1:
            boxes.append(((W - card_w) // 2, bot_y, card_w, bot_h))
        else:
            boxes.append((PAD_X, bot_y, card_w, bot_h))
            boxes.append((W - PAD_X - card_w, bot_y, card_w, bot_h))
        body_h = top_h + gap_y + bot_h

    legend_h = 34
    total_h = body_h + legend_h

    edges = list(diagram.character_edges or [])
    if not edges:
        from src.models import KoreanCharacterEdge
        rel_labels = list(diagram.relation_labels or [])
        for i in range(1, n):
            lbl = rel_labels[i - 1] if (i - 1) < len(rel_labels) else "관계"
            edges.append(KoreanCharacterEdge(source_idx=0, target_idx=i, relation=lbl, polarity="neutral"))

    polarity_style = {
        "positive": ((22, 163, 74), "+ "),
        "negative": ((220, 38, 38), "- "),
        "neutral": ((71, 85, 105), ""),
    }

    def _draw_fn(draw: ImageDraw.ImageDraw) -> None:
        for i, nd in enumerate(nodes):
            bx, by, bw, bh = boxes[i]
            _draw_node_card(draw, nd, bx, by, bw, bh, fonts["sub"], fonts["title"], fonts["body"])

        for idx_e, ed in enumerate(edges):
            if not (0 <= ed.source_idx < n and 0 <= ed.target_idx < n and ed.source_idx != ed.target_idx):
                continue
            x1, y1, w1, h1 = boxes[ed.source_idx]
            x2, y2, w2, h2 = boxes[ed.target_idx]
            c1x, c1y = x1 + w1 // 2, y1 + h1 // 2
            c2x, c2y = x2 + w2 // 2, y2 + h2 // 2

            # 박스 경계점 근사
            if abs(c1y - c2y) < 35:
                sx, sy = (x1 + w1 + 4, c1y) if c1x < c2x else (x1 - 4, c1y)
                tx, ty = (x2 - 4, c2y) if c1x < c2x else (x2 + w2 + 4, c2y)
            elif abs(c1x - c2x) < 35:
                sx, sy = (c1x, y1 + h1 + 4) if c1y < c2y else (c1x, y1 - 4)
                tx, ty = (c2x, y2 - 4) if c1y < c2y else (c2x, y2 + h2 + 4)
            else:
                sx = x1 + w1 if c1x < c2x else x1
                sy = y1 + h1 if c1y < c2y else y1
                tx = x2 if c1x < c2x else x2 + w2
                ty = y2 if c1y < c2y else y2 + h2

            color, prefix = polarity_style.get(ed.polarity, ((71, 85, 105), ""))
            raw_rel = (ed.relation or "").strip()
            lbl = f"{prefix}{raw_rel}" if raw_rel else ""
            # 대각선 교차 시 텍스트 겹침 완화용 Y 오프셋
            draw.line([(sx, sy), (tx, ty)], fill=color, width=3)
            angle = math.atan2(ty - sy, tx - sx)
            head_len = 13
            p1 = (tx + head_len * math.cos(angle + math.pi * 0.82), ty + head_len * math.sin(angle + math.pi * 0.82))
            p2 = (tx + head_len * math.cos(angle - math.pi * 0.82), ty + head_len * math.sin(angle - math.pi * 0.82))
            draw.polygon([(tx, ty), p1, p2], fill=color)
            if lbl:
                clean_lbl = " ".join(lbl.split())
                while len(clean_lbl) > 2 and draw.textlength(clean_lbl, font=fonts["lbl"]) > 110:
                    clean_lbl = clean_lbl[:-1].rstrip()
                mx = (sx + tx) // 2
                is_diag = abs(sx - tx) > 80 and abs(sy - ty) > 60
                if is_diag:
                    is_main_diag = (sx < tx) == (sy < ty)
                    mx += -75 if is_main_diag else 75
                    my = (sy + ty) // 2 + (-22 if is_main_diag else 22)
                else:
                    my = (sy + ty) // 2 - 16
                tw = int(draw.textlength(clean_lbl, font=fonts["lbl"])) + 14
                th = 22
                draw.rounded_rectangle(
                    [mx - tw // 2, my - th // 2, mx + tw // 2, my + th // 2],
                    radius=8,
                    fill=(255, 255, 255),
                    outline=color,
                    width=2,
                )
                draw.text((mx - tw // 2 + 7, my - th // 2 + 3), clean_lbl, font=fonts["lbl"], fill=color)

        # 하단 범례
        ly = body_top + body_h + 8
        draw.text((PAD_X, ly), "범례:  초록(+): 조력·지지    |    빨강(-): 갈등·대립    |    회색: 일반·혈연 관계", font=fonts["sub"], fill=(71, 85, 105))

    return total_h, [_draw_fn]


def _render_emotion_arc(
    diagram: KoreanVisualDiagram,
    W: int,
    PAD_X: int,
    body_top: int,
    fonts: dict[str, Any],
) -> tuple[int, list[Any]]:
    """문학 시·시가 시상 전개 · 정서 변화 그래프 (-2 ~ +2)."""
    pts = list(diagram.emotion_points or [])[:5]
    if not pts and diagram.nodes:
        from src.models import KoreanEmotionPoint
        pts = [
            KoreanEmotionPoint(
                stage=nd.subtitle or nd.title[:8],
                emotion=nd.details[0] if nd.details else nd.title,
                level=1 if i % 2 == 1 else -1,
            )
            for i, nd in enumerate(diagram.nodes[:5])
        ]
    while len(pts) < 2:
        from src.models import KoreanEmotionPoint
        pts.append(KoreanEmotionPoint(stage=f"구간 {len(pts) + 1}", emotion="정서 전개", level=0))

    plot_h = 340
    cards = list(diagram.nodes or [])[:4]
    probe = ImageDraw.Draw(Image.new("RGB", (W, 100)))
    cards_h = 0
    card_w = 0
    if cards:
        gap = 28
        card_w = (W - PAD_X * 2 - gap * (len(cards) - 1)) // len(cards)
        cards_h = max(_measure_card_height(probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"]) for nd in cards)
    total_h = plot_h + (20 + cards_h if cards else 0)

    def _draw_fn(draw: ImageDraw.ImageDraw) -> None:
        gx0 = PAD_X + 135
        gx1 = W - PAD_X - 45
        gy0 = body_top + 38
        gy1 = body_top + plot_h - 54
        gy_mid = (gy0 + gy1) // 2

        draw.rounded_rectangle(
            [PAD_X, body_top, W - PAD_X, body_top + plot_h],
            radius=14,
            fill=(248, 250, 252),
            outline=(203, 213, 225),
            width=2,
        )

        # 정서 극성 기준선 (+2, +1, 0, -1, -2)
        level_labels = [
            (2, "+2 긍정·극복", (22, 163, 74)),
            (1, "+1 희망·그리움", (34, 197, 94)),
            (0, " 0 기준·전환", (71, 85, 105)),
            (-1, "-1 아쉬움·고독", (217, 119, 6)),
            (-2, "-2 시련·절망", (220, 38, 38)),
        ]
        for lv, l_txt, l_col in level_labels:
            y_pos = int(gy_mid - (lv / 2.0) * ((gy1 - gy0) * 0.42))
            line_w = 2 if lv == 0 else 1
            line_c = (148, 163, 184) if lv == 0 else (226, 232, 240)
            draw.line([(gx0, y_pos), (gx1, y_pos)], fill=line_c, width=line_w)
            draw.text((PAD_X + 14, y_pos - 9), l_txt, font=fonts["lbl"], fill=l_col)

        # 세로축
        draw.line([(gx0, gy0 - 12), (gx0, gy1 + 10)], fill=(30, 41, 59), width=2)

        # 좌표 계산 및 꺾은선 연결
        n_pts = len(pts)
        coords: list[tuple[int, int, Any]] = []
        span_x = gx1 - gx0 - 110
        for i, pt in enumerate(pts):
            px = int(gx0 + 55 + (i / max(1, n_pts - 1)) * span_x)
            lv = max(-2, min(2, int(pt.level)))
            py = int(gy_mid - (lv / 2.0) * ((gy1 - gy0) * 0.42))
            coords.append((px, py, pt))

        if len(coords) >= 2:
            draw.line([(c[0], c[1]) for c in coords], fill=(37, 99, 235), width=5)

        for i, (px, py, pt) in enumerate(coords):
            lv = int(pt.level)
            dot_c = (22, 163, 74) if lv > 0 else ((220, 38, 38) if lv < 0 else (37, 99, 235))
            draw.ellipse([px - 10, py - 10, px + 10, py + 10], fill=dot_c, outline=(255, 255, 255), width=3)

            # 구간(stage) 하단 X축 라벨
            st_txt = " ".join((pt.stage or "").split())[:10]
            stw = draw.textlength(st_txt, font=fonts["sub"])
            draw.text((px - stw / 2, gy1 + 16), st_txt, font=fonts["sub"], fill=(15, 23, 42))

            # 정서(emotion) 배지
            em_txt = " ".join((pt.emotion or "").split())[:14]
            etw = int(draw.textlength(em_txt, font=fonts["lbl"])) + 16
            above = (lv >= 0) if lv != 0 else (i % 2 == 0)
            by = py - 38 if above else py + 16
            draw.rounded_rectangle(
                [px - etw // 2, by, px + etw // 2, by + 24],
                radius=8,
                fill=(255, 255, 255),
                outline=dot_c,
                width=2,
            )
            draw.text((px - etw // 2 + 8, by + 4), em_txt, font=fonts["lbl"], fill=dot_c)

        if cards:
            cy = body_top + plot_h + 20
            gap = 28
            for i, nd in enumerate(cards):
                _draw_node_card(draw, nd, PAD_X + i * (card_w + gap), cy, card_w, cards_h, fonts["sub"], fonts["title"], fonts["body"])

    return total_h, [_draw_fn]


def _render_poetic_contrast(
    diagram: KoreanVisualDiagram,
    W: int,
    PAD_X: int,
    body_top: int,
    fonts: dict[str, Any],
) -> tuple[int, list[Any]]:
    """화자 중심 (+) 지향 시어 vs (-) 대립 시어 이분법 대조 도식."""
    nodes = list(diagram.nodes or [])[:2]
    while len(nodes) < 2:
        nodes.append(KoreanDiagramNode(title="(+) 긍정 시어" if len(nodes) == 0 else "(-) 부정 시어"))
    nodes[0].accent_color = nodes[0].accent_color or "green"
    nodes[1].accent_color = "amber" if nodes[1].accent_color in ("blue", "green") else nodes[1].accent_color

    speaker_txt = (diagram.center_label or "시적 화자 · 내적 태도").strip()
    rel_labels = list(diagram.relation_labels or [])
    probe = ImageDraw.Draw(Image.new("RGB", (W, 100)))

    spk_w = min(560, W - PAD_X * 2)
    spk_lines = _wrap_text(probe, speaker_txt, fonts["title"], spk_w - 40)
    spk_h = max(78, 36 + len(spk_lines) * 28 + 14)

    card_w = 570
    card_h = max(_measure_card_height(probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"]) for nd in nodes)
    gap_y = 66
    total_h = spk_h + gap_y + card_h

    def _draw_fn(draw: ImageDraw.ImageDraw) -> None:
        cx = W // 2
        sx = (W - spk_w) // 2
        draw.rounded_rectangle(
            [sx, body_top, sx + spk_w, body_top + spk_h],
            radius=18,
            fill=(239, 246, 255),
            outline=(29, 78, 216),
            width=4,
        )
        draw.text((sx + 20, body_top + 10), "시적 화자 · 중심 상황", font=fonts["sub"], fill=(29, 78, 216))
        sy_t = body_top + 38
        for sl in spk_lines:
            tw = draw.textlength(sl, font=fonts["title"])
            draw.text((cx - tw / 2, sy_t), sl, font=fonts["title"], fill=(15, 23, 42))
            sy_t += 28

        mid_y = body_top + spk_h + gap_y // 2
        card_y = body_top + spk_h + gap_y
        lx, rx = PAD_X, W - PAD_X - card_w
        l_cx, r_cx = lx + card_w // 2, rx + card_w // 2

        draw.line([(cx, body_top + spk_h), (cx, mid_y)], fill=(71, 85, 105), width=3)
        draw.line([(l_cx, mid_y), (r_cx, mid_y)], fill=(71, 85, 105), width=3)

        l_lbl = rel_labels[0] if len(rel_labels) >= 1 and rel_labels[0].strip() else "(+) 지향·긍정"
        r_lbl = rel_labels[1] if len(rel_labels) >= 2 and rel_labels[1].strip() else "(-) 대립·부정"
        _draw_arrow(draw, l_cx, mid_y, l_cx, card_y - 4, label=l_lbl, f_lbl=fonts["lbl"], color=(22, 163, 74))
        _draw_arrow(draw, r_cx, mid_y, r_cx, card_y - 4, label=r_lbl, f_lbl=fonts["lbl"], color=(217, 119, 6))

        _draw_node_card(draw, nodes[0], lx, card_y, card_w, card_h, fonts["sub"], fonts["title"], fonts["body"])
        _draw_node_card(draw, nodes[1], rx, card_y, card_w, card_h, fonts["sub"], fonts["title"], fonts["body"])

    return total_h, [_draw_fn]


def _render_space_journey(
    diagram: KoreanVisualDiagram,
    W: int,
    PAD_X: int,
    body_top: int,
    fonts: dict[str, Any],
) -> tuple[int, list[Any]]:
    """문학 공간 · 시선 이동 경로도: 이동 트랙 핀 + 구간 화살표 + 공간별 상징·정서 카드."""
    nodes = list(diagram.nodes or [])[:4]
    while len(nodes) < 2:
        nodes.append(KoreanDiagramNode(title=f"공간 {len(nodes) + 1}"))
    n = len(nodes)
    rel_labels = list(diagram.relation_labels or [])
    probe = ImageDraw.Draw(Image.new("RGB", (W, 100)))

    gap = 34
    card_w = (W - PAD_X * 2 - gap * (n - 1)) // n
    card_h = max(_measure_card_height(probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"]) for nd in nodes)
    track_h = 64
    total_h = track_h + card_h

    def _draw_fn(draw: ImageDraw.ImageDraw) -> None:
        track_y = body_top + 26
        card_y = body_top + track_h
        centers = [PAD_X + i * (card_w + gap) + card_w // 2 for i in range(n)]

        for i in range(n - 1):
            lbl = rel_labels[i] if i < len(rel_labels) and rel_labels[i].strip() else "이동"
            _draw_arrow(draw, centers[i] + 22, track_y, centers[i + 1] - 22, track_y, label=lbl, f_lbl=fonts["lbl"], color=(37, 99, 235))

        for i, nd in enumerate(nodes):
            cx = centers[i]
            pal = _PALETTES.get(nd.accent_color or _COLOR_CYCLE[i % len(_COLOR_CYCLE)], _PALETTES["blue"])
            draw.line([(cx, track_y + 18), (cx, card_y)], fill=pal["border"], width=3)
            draw.ellipse([cx - 18, track_y - 18, cx + 18, track_y + 18], fill=pal["hdr"], outline=(255, 255, 255), width=3)
            pin_s = str(i + 1)
            pw = draw.textlength(pin_s, font=fonts["sub"])
            draw.text((cx - pw / 2, track_y - 9), pin_s, font=fonts["sub"], fill=(255, 255, 255))
            _draw_node_card(draw, nd, PAD_X + i * (card_w + gap), card_y, card_w, card_h, fonts["sub"], fonts["title"], fonts["body"])

    return total_h, [_draw_fn]


def _render_card_layout(
    diagram: KoreanVisualDiagram,
    dtype: str,
    W: int,
    PAD_X: int,
    body_top: int,
    fonts: dict[str, Any],
) -> tuple[int, list[Any]]:
    """flow / comparison / hierarchy / 원형 cycle 레이아웃."""
    nodes = list(diagram.nodes or [])[:5]
    n = len(nodes)
    rel_labels = list(diagram.relation_labels or [])
    probe = ImageDraw.Draw(Image.new("RGB", (W, 100)))
    GAP_X = 88
    GAP_Y = 48

    boxes: list[tuple[KoreanDiagramNode, int, int, int, int]] = []
    body_h = 0

    if dtype == "hierarchy" and n >= 2:
        root_w = min(680, W - PAD_X * 2)
        root_x = (W - root_w) // 2
        root_h = _measure_card_height(probe, nodes[0], root_w, fonts["sub"], fonts["title"], fonts["body"])
        boxes.append((nodes[0], root_x, body_top, root_w, root_h))
        children = nodes[1:]
        cn = len(children)
        child_w = (W - PAD_X * 2 - GAP_X * (cn - 1)) // cn
        child_h = max(_measure_card_height(probe, c, child_w, fonts["sub"], fonts["title"], fonts["body"]) for c in children)
        child_y = body_top + root_h + 56
        for i, c_node in enumerate(children):
            boxes.append((c_node, PAD_X + i * (child_w + GAP_X), child_y, child_w, child_h))
        body_h = root_h + 56 + child_h

    elif dtype == "cycle" and n >= 3:
        card_w = (W - PAD_X * 2 - 120) // 2
        top_nodes = nodes[:2]
        bot_nodes = nodes[2:4]
        top_h = max(_measure_card_height(probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"]) for nd in top_nodes)
        bot_h = max(_measure_card_height(probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"]) for nd in bot_nodes)
        boxes.append((top_nodes[0], PAD_X, body_top, card_w, top_h))
        boxes.append((top_nodes[1], W - PAD_X - card_w, body_top, card_w, top_h))
        bot_y = body_top + top_h + 76
        if len(bot_nodes) == 1:
            boxes.append((bot_nodes[0], (W - card_w) // 2, bot_y, card_w, bot_h))
        else:
            boxes.append((bot_nodes[1], PAD_X, bot_y, card_w, bot_h))
            boxes.append((bot_nodes[0], W - PAD_X - card_w, bot_y, card_w, bot_h))
        body_h = top_h + 76 + bot_h

    else:
        if n <= 4:
            gap = 74 if dtype == "comparison" else GAP_X
            card_w = (W - PAD_X * 2 - gap * (n - 1)) // n
            card_h = max(_measure_card_height(probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"]) for nd in nodes)
            for i, nd in enumerate(nodes):
                boxes.append((nd, PAD_X + i * (card_w + gap), body_top, card_w, card_h))
            body_h = card_h
        else:
            top_nodes = nodes[:3]
            bot_nodes = nodes[3:]
            card_w = (W - PAD_X * 2 - GAP_X * 2) // 3
            top_h = max(_measure_card_height(probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"]) for nd in top_nodes)
            bot_h = max(_measure_card_height(probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"]) for nd in bot_nodes)
            for i, nd in enumerate(top_nodes):
                boxes.append((nd, PAD_X + i * (card_w + GAP_X), body_top, card_w, top_h))
            bot_y = body_top + top_h + GAP_Y + 12
            bot_start_x = (W - (len(bot_nodes) * card_w + (len(bot_nodes) - 1) * GAP_X)) // 2
            for i, nd in enumerate(bot_nodes):
                boxes.append((nd, bot_start_x + i * (card_w + GAP_X), bot_y, card_w, bot_h))
            body_h = top_h + GAP_Y + 12 + bot_h

    def _draw_fn(draw: ImageDraw.ImageDraw) -> None:
        for nd, bx, by, bw, bh in boxes:
            _draw_node_card(draw, nd, bx, by, bw, bh, fonts["sub"], fonts["title"], fonts["body"])

        if dtype == "hierarchy" and len(boxes) >= 2:
            _, rx, ry, rw, rh = boxes[0]
            root_cx, root_by = rx + rw // 2, ry + rh
            mid_y = root_by + 28
            draw.line([(root_cx, root_by), (root_cx, mid_y)], fill=(71, 85, 105), width=3)
            child_centers = [(cx + cw // 2, cy) for _, cx, cy, cw, _ in boxes[1:]]
            if len(child_centers) >= 2:
                draw.line([(child_centers[0][0], mid_y), (child_centers[-1][0], mid_y)], fill=(71, 85, 105), width=3)
            for idx_c, (ccx, ccy) in enumerate(child_centers):
                lbl = rel_labels[idx_c] if idx_c < len(rel_labels) else ""
                _draw_arrow(draw, ccx, mid_y, ccx, ccy - 4, label=lbl, f_lbl=fonts["lbl"])

        elif dtype == "cycle" and len(boxes) >= 3:
            # 중앙 순환 배지
            c_txt = (diagram.center_label or "순환 · 상호작용").strip()[:14]
            cw = max(120, int(draw.textlength(c_txt, font=fonts["sub"])) + 28)
            cy_mid = body_top + body_h // 2
            draw.rounded_rectangle(
                [W // 2 - cw // 2, cy_mid - 20, W // 2 + cw // 2, cy_mid + 20],
                radius=16,
                fill=(239, 246, 255),
                outline=(59, 130, 246),
                width=2,
            )
            draw.text((W // 2 - cw // 2 + 14, cy_mid - 10), c_txt, font=fonts["sub"], fill=(29, 78, 216))

            for i in range(len(boxes)):
                _, x1, y1, w1, h1 = boxes[i]
                _, x2, y2, w2, h2 = boxes[(i + 1) % len(boxes)]
                lbl = rel_labels[i] if i < len(rel_labels) else ""
                c1x, c1y = x1 + w1 // 2, y1 + h1 // 2
                c2x, c2y = x2 + w2 // 2, y2 + h2 // 2
                if abs(c1y - c2y) < 40:
                    if c1x < c2x:
                        _draw_arrow(draw, x1 + w1 + 4, c1y, x2 - 4, c2y, label=lbl, f_lbl=fonts["lbl"])
                    else:
                        _draw_arrow(draw, x1 - 4, c1y, x2 + w2 + 4, c2y, label=lbl, f_lbl=fonts["lbl"])
                elif c1y < c2y:
                    _draw_arrow(draw, c1x, y1 + h1 + 4, c2x, y2 - 4, label=lbl, f_lbl=fonts["lbl"])
                else:
                    _draw_arrow(draw, c1x, y1 - 4, c2x, y2 + h2 + 4, label=lbl, f_lbl=fonts["lbl"])

        elif dtype == "comparison":
            for i in range(len(boxes) - 1):
                _, x1, y1, w1, h1 = boxes[i]
                _, x2, _, _, _ = boxes[i + 1]
                mx = (x1 + w1 + x2) // 2
                my = y1 + h1 // 2
                vs_lbl = " ".join((rel_labels[i] if i < len(rel_labels) else "VS").split()).strip() or "VS"
                while len(vs_lbl) > 2 and draw.textlength(vs_lbl, font=fonts["lbl"]) > 58:
                    vs_lbl = vs_lbl[:-1].rstrip()
                tw = max(36, int(draw.textlength(vs_lbl, font=fonts["lbl"])) + 16)
                draw.rounded_rectangle(
                    [mx - tw // 2, my - 15, mx + tw // 2, my + 15],
                    radius=11,
                    fill=(239, 246, 255),
                    outline=(59, 130, 246),
                    width=2,
                )
                draw.text((mx - tw // 2 + 8, my - 8), vs_lbl, font=fonts["lbl"], fill=(29, 78, 216))

        else:
            for i in range(len(boxes) - 1):
                _, x1, y1, w1, h1 = boxes[i]
                _, x2, y2, w2, h2 = boxes[i + 1]
                lbl = rel_labels[i] if i < len(rel_labels) else ""
                if abs(y1 - y2) < 40:
                    _draw_arrow(draw, x1 + w1 + 4, y1 + h1 // 2, x2 - 4, y2 + h2 // 2, label=lbl, f_lbl=fonts["lbl"])
                else:
                    _draw_arrow(draw, x1 + w1 // 2, y1 + h1 + 4, x2 + w2 // 2, y2 - 4, label=lbl, f_lbl=fonts["lbl"])

    return body_h, [_draw_fn]


def render_korean_diagram_png(diagram: KoreanVisualDiagram) -> bytes:
    """KoreanVisualDiagram 스펙을 16종(비문학 12종+문학 4종) 고해상도 한글 도식 PNG로 렌더링한다."""
    dtype = (diagram.diagram_type or "flow").strip().lower()
    if not (diagram.nodes or []) and dtype not in ("graph", "emotion_arc"):
        raise ValueError("도식에 표시할 노드가 없습니다.")

    W = 1400
    PAD_X = 36
    fonts = {
        "hdr_tag": _load_font(17, bold=True),
        "hdr_title": _load_font(26, bold=True),
        "sub": _load_font(16, bold=True),
        "title": _load_font(21, bold=True),
        "body": _load_font(18, bold=False),
        "lbl": _load_font(14, bold=True),
        "take": _load_font(19, bold=True),
    }

    probe = ImageDraw.Draw(Image.new("RGB", (W, 200), (255, 255, 255)))
    title_lines = _wrap_text(probe, diagram.title or "핵심 원리 도식", fonts["hdr_title"], W - PAD_X * 2 - 20)
    header_h = 26 + 26 + max(1, len(title_lines)) * 34 + 16
    body_top = header_h + 18

    takeaway_lines = (
        _wrap_text(probe, f"핵심 정리: {diagram.takeaway}", fonts["take"], W - PAD_X * 2 - 44)
        if (diagram.takeaway or "").strip()
        else []
    )
    takeaway_h = (len(takeaway_lines) * 28 + 28) if takeaway_lines else 0

    if dtype == "matrix":
        body_h, draw_fns = _render_matrix(diagram, W, PAD_X, body_top, fonts)
    elif dtype == "graph":
        body_h, draw_fns = _render_graph(diagram, W, PAD_X, body_top, fonts)
    elif dtype == "anatomy":
        body_h, draw_fns = _render_anatomy(diagram, W, PAD_X, body_top, fonts)
    elif dtype == "timeline":
        body_h, draw_fns = _render_timeline(diagram, W, PAD_X, body_top, fonts)
    elif dtype == "venn":
        body_h, draw_fns = _render_venn(diagram, W, PAD_X, body_top, fonts)
    elif dtype == "scale":
        body_h, draw_fns = _render_scale(diagram, W, PAD_X, body_top, fonts)
    elif dtype == "fishbone":
        body_h, draw_fns = _render_fishbone(diagram, W, PAD_X, body_top, fonts)
    elif dtype == "quadrant":
        body_h, draw_fns = _render_quadrant(diagram, W, PAD_X, body_top, fonts)
    elif dtype == "decision_tree":
        body_h, draw_fns = _render_decision_tree(diagram, W, PAD_X, body_top, fonts)
    elif dtype == "pyramid":
        body_h, draw_fns = _render_pyramid(diagram, W, PAD_X, body_top, fonts)
    elif dtype == "character_map":
        body_h, draw_fns = _render_character_map(diagram, W, PAD_X, body_top, fonts)
    elif dtype == "emotion_arc":
        body_h, draw_fns = _render_emotion_arc(diagram, W, PAD_X, body_top, fonts)
    elif dtype == "poetic_contrast":
        body_h, draw_fns = _render_poetic_contrast(diagram, W, PAD_X, body_top, fonts)
    elif dtype == "space_journey":
        body_h, draw_fns = _render_space_journey(diagram, W, PAD_X, body_top, fonts)
    else:
        body_h, draw_fns = _render_card_layout(diagram, dtype, W, PAD_X, body_top, fonts)

    total_h = body_top + body_h + 24 + (takeaway_h + 18 if takeaway_h else 0)
    img = Image.new("RGB", (W, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle(
        [8, 8, W - 9, total_h - 9],
        radius=18,
        fill=(255, 255, 255),
        outline=(203, 213, 225),
        width=3,
    )
    type_tag_map = {
        "flow": "원리 · 과정 흐름도",
        "matrix": "항목별 정밀 비교 매트릭스표",
        "graph": "핵심 변수 관계 그래프",
        "anatomy": "구조 · 장치 원리 모식도",
        "cycle": "상호작용 · 순환 피드백 루프",
        "timeline": "시대 · 사상 변천 연표",
        "venn": "개념 포함 · 교집합 벤다이어그램",
        "scale": "가치 대립 · 균형 저울도",
        "fishbone": "다발적 원인 → 결과 어골도",
        "quadrant": "2×2 사분면 분류 도식",
        "decision_tree": "조건 분기 · 판정 순서도",
        "pyramid": "위계 · 계층 피라미드",
        "character_map": "소설 인물관계도 · 갈등 구조도",
        "emotion_arc": "시상 전개 · 정서 변화 그래프",
        "poetic_contrast": "화자 중심 (+) / (-) 시어 대립도",
        "space_journey": "공간 · 시선 이동 경로도",
        "comparison": "개념 · 이론 비교 도식",
        "hierarchy": "개념 체계 · 분류 구조도",
    }
    tag_text = type_tag_map.get(dtype, "개념 이해 도식")
    draw.text((PAD_X, 24), f"◆ {tag_text}", font=fonts["hdr_tag"], fill=(29, 78, 216))
    ty = 52
    for tl in title_lines:
        draw.text((PAD_X, ty), tl, font=fonts["hdr_title"], fill=(15, 23, 42))
        ty += 34
    draw.line([(PAD_X, header_h - 4), (W - PAD_X, header_h - 4)], fill=(226, 232, 240), width=2)

    for fn in draw_fns:
        fn(draw)

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
            draw.text((PAD_X + 20, cur_ty), tl, font=fonts["take"], fill=(146, 64, 14))
            cur_ty += 28

    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()
