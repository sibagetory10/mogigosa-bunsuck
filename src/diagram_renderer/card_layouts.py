from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageDraw

from src.diagram_renderer.academic_charts import (
    _render_anatomy,
    _render_decision_tree,
    _render_fishbone,
    _render_graph,
    _render_matrix,
    _render_pyramid,
    _render_quadrant,
    _render_scale,
    _render_timeline,
    _render_venn,
)
from src.diagram_renderer.literature_charts import (
    _render_character_map,
    _render_emotion_arc,
    _render_poetic_contrast,
    _render_space_journey,
)
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

from src.models.korean import KoreanDiagramNode

if TYPE_CHECKING:
    from src.models import KoreanVisualDiagram

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


def _render_compact_vertical_layout(
    diagram: KoreanVisualDiagram,
    dtype: str,
    W: int,
    PAD_X: int,
    body_top: int,
    fonts: dict[str, Any],
) -> tuple[int, list[Any]]:
    """좁은 컬럼(영어 분석지 좌측 등)에서 글씨가 큼직하게 읽히도록 세로 중심 고가독성 레이아웃을 계산한다."""
    probe = ImageDraw.Draw(Image.new("RGB", (W, 100)))
    inner_full_w = W - PAD_X * 2
    nodes = list(diagram.nodes or [])[:6]
    rel_labels = list(diagram.relation_labels or [])

    # 1) 비교 매트릭스표 (760px 폭 + 큰 폰트)
    if dtype == "matrix" and (nodes or diagram.matrix_rows):
        m_nodes = nodes[:3] if len(nodes) >= 2 else (nodes + [KoreanDiagramNode(title="대상 B")])[:2]
        col_cnt = len(m_nodes)
        rows = list(diagram.matrix_rows or [])
        if not rows:
            max_d = max((len(nd.details or []) for nd in m_nodes), default=1)
            for ri in range(max(1, min(4, max_d))):
                rows.append(
                    type("_Row", (), {
                        "criterion": f"기준 {ri + 1}",
                        "values": [
                            (nd.details[ri] if ri < len(nd.details or []) else "-")
                            for nd in m_nodes
                        ],
                    })()
                )
        crit_w = 152
        val_w = (inner_full_w - crit_w) // col_cnt
        hdr_h = 52
        row_heights: list[int] = []
        for r in rows[:5]:
            c_lines = _wrap_text(probe, r.criterion, fonts["sub"], crit_w - 18)
            max_l = len(c_lines)
            for ci in range(col_cnt):
                vt = r.values[ci] if ci < len(r.values) else "-"
                max_l = max(max_l, len(_wrap_text(probe, vt, fonts["body"], val_w - 18)))
            row_heights.append(max(48, max_l * 27 + 18))
        total_h = hdr_h + sum(row_heights) + 6

        def _draw_compact_matrix(draw: ImageDraw.ImageDraw) -> None:
            x0, y0 = PAD_X, body_top
            draw.rounded_rectangle([x0, y0, x0 + crit_w - 2, y0 + hdr_h], radius=8, fill=(30, 41, 59))
            draw.text((x0 + 14, y0 + 14), "비교 기준", font=fonts["sub"], fill=(255, 255, 255))
            for ci, nd in enumerate(m_nodes):
                pal = _PALETTES.get(nd.accent_color or _COLOR_CYCLE[ci % len(_COLOR_CYCLE)], _PALETTES["blue"])
                cx = x0 + crit_w + ci * val_w + 2
                draw.rounded_rectangle([cx, y0, cx + val_w - 4, y0 + hdr_h], radius=8, fill=pal["hdr"])
                tl = _wrap_text(draw, nd.title, fonts["sub"], val_w - 16)
                draw.text((cx + 10, y0 + (14 if len(tl) == 1 else 6)), tl[0], font=fonts["sub"], fill=(255, 255, 255))
                if len(tl) > 1:
                    draw.text((cx + 10, y0 + 27), tl[1], font=fonts["sub"], fill=(255, 255, 255))
            cur_y = y0 + hdr_h + 4
            for ri, r in enumerate(rows[:5]):
                rh = row_heights[ri]
                bg_c = (248, 250, 252) if ri % 2 == 0 else (255, 255, 255)
                draw.rectangle([x0, cur_y, x0 + inner_full_w, cur_y + rh], fill=bg_c, outline=(203, 213, 225), width=2)
                draw.rectangle([x0, cur_y, x0 + crit_w, cur_y + rh], fill=(241, 245, 249), outline=(203, 213, 225), width=2)
                cy_t = cur_y + 10
                for cl in _wrap_text(draw, r.criterion, fonts["sub"], crit_w - 18):
                    draw.text((x0 + 10, cy_t), cl, font=fonts["sub"], fill=(30, 58, 138))
                    cy_t += 26
                for ci in range(col_cnt):
                    cx = x0 + crit_w + ci * val_w
                    draw.line([(cx, cur_y), (cx, cur_y + rh)], fill=(203, 213, 225), width=2)
                    vt = r.values[ci] if ci < len(r.values) else "-"
                    vy = cur_y + 10
                    for vl in _wrap_text(draw, vt, fonts["body"], val_w - 18):
                        draw.text((cx + 10, vy), vl, font=fonts["body"], fill=(30, 41, 59))
                        vy += 26
                cur_y += rh

        return total_h, [_draw_compact_matrix]

    # 2) 2자 대립/비교/저울/벤다이어그램 (노드 2개 → 2열 큰 카드)
    if dtype in ("comparison", "scale", "venn") and len(nodes) == 2:
        gap = 36
        card_w = (inner_full_w - gap) // 2
        card_h = max(
            _measure_card_height(
                probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"],
                sub_h=30, title_lh=30, body_lh=27, min_h=120,
            )
            for nd in nodes
        )
        c_lbl = (diagram.center_label or "").strip()
        extra_h = 52 if (dtype == "venn" and c_lbl) else 0
        total_h = card_h + extra_h

        def _draw_compact_pair(draw: ImageDraw.ImageDraw) -> None:
            _draw_node_card(
                draw, nodes[0], PAD_X, body_top, card_w, card_h,
                fonts["sub"], fonts["title"], fonts["body"], sub_h=30, title_lh=30, body_lh=27,
            )
            _draw_node_card(
                draw, nodes[1], PAD_X + card_w + gap, body_top, card_w, card_h,
                fonts["sub"], fonts["title"], fonts["body"], sub_h=30, title_lh=30, body_lh=27,
            )
            mx = PAD_X + card_w + gap // 2
            my = body_top + card_h // 2
            vs_txt = (rel_labels[0] if rel_labels and rel_labels[0].strip() else "VS")[:6]
            tw = max(44, int(draw.textlength(vs_txt, font=fonts["lbl"])) + 16)
            draw.rounded_rectangle(
                [mx - tw // 2, my - 18, mx + tw // 2, my + 18],
                radius=12, fill=(239, 246, 255), outline=(59, 130, 246), width=2,
            )
            draw.text((mx - tw // 2 + 8, my - 10), vs_txt, font=fonts["lbl"], fill=(29, 78, 216))
            if dtype == "venn" and c_lbl:
                cy = body_top + card_h + 12
                draw.rounded_rectangle(
                    [PAD_X, cy, W - PAD_X, cy + 38],
                    radius=10, fill=(238, 242, 255), outline=(99, 102, 241), width=2,
                )
                txt = f"공통점: {c_lbl}"
                draw.text((PAD_X + 16, cy + 8), txt, font=fonts["sub"], fill=(55, 48, 163))

        return total_h, [_draw_compact_pair]

    # 3) 기본 세로 흐름 스택 (flow / timeline / hierarchy / cycle / anatomy / fishbone / pyramid / decision_tree 등)
    if not nodes:
        nodes = [KoreanDiagramNode(title=diagram.title or "핵심 구조", details=[diagram.takeaway] if diagram.takeaway else [])]

    show_center_banner = (
        dtype in ("anatomy", "cycle", "fishbone", "decision_tree")
        and bool((diagram.center_label or "").strip())
    )
    banner_h = 46 if show_center_banner else 0
    arrow_gap = 38
    card_w = inner_full_w
    eff_title_lh = _measure_line_height(fonts["title"], 30)
    eff_body_lh = _measure_line_height(fonts["body"], 27)
    card_hs = [
        _measure_card_height(
            probe, nd, card_w, fonts["sub"], fonts["title"], fonts["body"],
            sub_h=30, title_lh=eff_title_lh, body_lh=eff_body_lh, min_h=88,
        ) + 6
        for nd in nodes
    ]
    total_h = (banner_h + 14 if show_center_banner else 0) + sum(card_hs) + arrow_gap * max(0, len(nodes) - 1) + 8

    def _draw_compact_stack(draw: ImageDraw.ImageDraw) -> None:
        cur_y = body_top
        if show_center_banner:
            c_txt = (diagram.center_label or "").strip()
            draw.rounded_rectangle(
                [PAD_X, cur_y, W - PAD_X, cur_y + banner_h],
                radius=12, fill=(239, 246, 255), outline=(37, 99, 235), width=2,
            )
            prefix = "핵심 중심 · " if dtype != "decision_tree" else "분기 조건 · "
            lbl_full = f"{prefix}{c_txt}"
            tw = draw.textlength(lbl_full, font=fonts["sub"])
            draw.text(((W - tw) / 2, cur_y + 12), lbl_full, font=fonts["sub"], fill=(29, 78, 216))
            cur_y += banner_h + 14

        cx = W // 2
        for i, nd in enumerate(nodes):
            ch = card_hs[i]
            _draw_node_card(
                draw, nd, PAD_X, cur_y, card_w, ch,
                fonts["sub"], fonts["title"], fonts["body"],
                sub_h=30, title_lh=eff_title_lh, body_lh=eff_body_lh,
            )
            if i < len(nodes) - 1:
                y1 = cur_y + ch + 3
                y2 = cur_y + ch + arrow_gap - 4
                draw.line([(cx, y1), (cx, y2)], fill=(71, 85, 105), width=3)
                draw.polygon([(cx, y2 + 2), (cx - 8, y2 - 10), (cx + 8, y2 - 10)], fill=(71, 85, 105))
                lbl = " ".join((rel_labels[i] if i < len(rel_labels) else "").split()).strip()
                if lbl:
                    tw = int(draw.textlength(lbl, font=fonts["lbl"])) + 18
                    ly_mid = (y1 + y2) // 2
                    draw.rounded_rectangle(
                        [cx + 16, ly_mid - 14, cx + 16 + tw, ly_mid + 14],
                        radius=8, fill=(255, 255, 255), outline=(148, 163, 184), width=2,
                    )
                    draw.text((cx + 25, ly_mid - 10), lbl, font=fonts["lbl"], fill=(30, 58, 138))
            cur_y += ch + arrow_gap

    return total_h, [_draw_compact_stack]


def render_korean_diagram_png(
    diagram: KoreanVisualDiagram,
    *,
    compact_vertical: bool = False,
) -> bytes:
    """KoreanVisualDiagram 스펙을 16종(비문학 12종+문학 4종) 고해상도 한글 도식 PNG로 렌더링한다.

    compact_vertical=True이면 영어 정밀 분석지 좌측 컬럼 등 좁은 폭에서도 글씨가 크게 읽히도록
    760px 세로 중심 고가독성 캔버스로 렌더링한다.
    """
    dtype = (diagram.diagram_type or "flow").strip().lower()
    if not (diagram.nodes or []) and dtype not in ("graph", "emotion_arc"):
        raise ValueError("도식에 표시할 노드가 없습니다.")

    if compact_vertical and dtype not in ("graph", "emotion_arc"):
        W = 760
        PAD_X = 22
        fonts = {
            "hdr_tag": _load_font(19, bold=True),
            "hdr_title": _load_font(26, bold=True),
            "sub": _load_font(18, bold=True),
            "title": _load_font(23, bold=True),
            "body": _load_font(20, bold=False),
            "lbl": _load_font(17, bold=True),
            "take": _load_font(20, bold=True),
        }
        title_lh = _measure_line_height(fonts["hdr_title"], 34)
        take_lh = _measure_line_height(fonts["take"], 30)
    else:
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
        title_lh = _measure_line_height(fonts["hdr_title"], 34)
        take_lh = _measure_line_height(fonts["take"], 30)

    probe = ImageDraw.Draw(Image.new("RGB", (W, 200), (255, 255, 255)))
    title_lines = _wrap_text(probe, diagram.title or "핵심 원리 도식", fonts["hdr_title"], W - PAD_X * 2 - 20)
    header_h = 24 + 28 + max(1, len(title_lines)) * title_lh + 16
    body_top = header_h + 16

    takeaway_lines = (
        _wrap_text(probe, f"핵심 정리: {diagram.takeaway}", fonts["take"], W - PAD_X * 2 - 36)
        if (diagram.takeaway or "").strip()
        else []
    )
    takeaway_pad_v = 14
    takeaway_h = (len(takeaway_lines) * take_lh + takeaway_pad_v * 2 + 4) if takeaway_lines else 0

    if compact_vertical and dtype not in ("graph", "emotion_arc"):
        body_h, draw_fns = _render_compact_vertical_layout(diagram, dtype, W, PAD_X, body_top, fonts)
    elif dtype == "matrix":
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

    ty_box = body_top + body_h + 18
    bottom_margin = 26
    total_h = (ty_box + takeaway_h + bottom_margin) if takeaway_lines else (body_top + body_h + bottom_margin + 4)
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
    draw.text((PAD_X, 22), f"◆ {tag_text}", font=fonts["hdr_tag"], fill=(29, 78, 216))
    ty = 52
    for tl in title_lines:
        draw.text((PAD_X, ty), tl, font=fonts["hdr_title"], fill=(15, 23, 42))
        ty += title_lh
    draw.line([(PAD_X, header_h - 4), (W - PAD_X, header_h - 4)], fill=(226, 232, 240), width=2)

    for fn in draw_fns:
        fn(draw)

    if takeaway_lines:
        draw.rounded_rectangle(
            [PAD_X, ty_box, W - PAD_X, ty_box + takeaway_h],
            radius=12,
            fill=(255, 251, 235),
            outline=(245, 158, 11),
            width=2,
        )
        cur_ty = ty_box + takeaway_pad_v
        for tl in takeaway_lines:
            draw.text((PAD_X + 18, cur_ty), tl, font=fonts["take"], fill=(146, 64, 14))
            cur_ty += take_lh

    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()
