from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageDraw

from src.diagram_renderer.primitives import (
    _COLOR_CYCLE,
    _PALETTES,
    _draw_arrow,
    _draw_node_card,
    _load_font,
    _measure_card_height,
    _measure_line_height,
    _wrap_text,
)

from src.models.korean import KoreanDiagramNode, KoreanGraphCurve

if TYPE_CHECKING:
    from src.models import KoreanVisualDiagram

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
