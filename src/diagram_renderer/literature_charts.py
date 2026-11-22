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

from src.models.korean import KoreanCharacterEdge, KoreanDiagramNode, KoreanEmotionPoint

if TYPE_CHECKING:
    from src.models import KoreanVisualDiagram

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
