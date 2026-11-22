from __future__ import annotations

import html
import io
import math
import os
from typing import Any

import pymupdf
from PIL import Image

from src.pdf.common import _CIRCLED_NUMS, _pick_korean_font_paths

def _measure_rendered_line_chars(
    text: str,
    fontfile: str | None,
    fontsize: float,
) -> tuple[list[tuple[float, float]], float]:
    """PyMuPDF 실제 렌더링 좌표(rawdict)로 각 글자의 (x0, x1) 오프셋과 전체 폭을 정확히 측정한다."""
    if not text:
        return [], 0.0
    tmp_doc = pymupdf.open()
    try:
        p = tmp_doc.new_page(width=2000, height=120)
        if fontfile:
            p.insert_font(fontname="mfont", fontfile=fontfile)
            fn = "mfont"
        else:
            fn = "korea"
        p.insert_text((0.0, 60.0), text, fontname=fn, fontsize=fontsize, color=(0, 0, 0))
        rd = p.get_text("rawdict")
        chars_bbox: list[tuple[float, float]] = []
        for blk in rd.get("blocks", []):
            for ln in blk.get("lines", []):
                for sp in ln.get("spans", []):
                    for ch in sp.get("chars", []):
                        bb = ch.get("bbox")
                        if bb:
                            chars_bbox.append((float(bb[0]), float(bb[2])))
        if len(chars_bbox) == len(text):
            return chars_bbox, chars_bbox[-1][1]
        # fallback: 공백 등 rawdict에서 누락된 문자가 있을 때 선형 보간
        f_obj = pymupdf.Font(fontfile=fontfile) if fontfile else pymupdf.Font("korea")
        space_w = fontsize if not fontfile else f_obj.text_length(" ", fontsize=fontsize)
        out: list[tuple[float, float]] = []
        cx = 0.0
        for ch in text:
            cw = space_w if ch == " " else f_obj.text_length(ch, fontsize=fontsize)
            out.append((cx, cx + cw))
            cx += cw
        return out, cx
    finally:
        tmp_doc.close()


def _wrap_text_by_width(text: str, font_obj: pymupdf.Font, fontsize: float, max_width: float) -> list[str]:
    """주어진 폭(pt)에 맞게 한글/영문 문자열을 줄바꿈한다."""
    s = (text or "").strip()
    if not s:
        return []
    lines: list[str] = []
    cur = ""
    for ch in s:
        if ch == "\n":
            if cur:
                lines.append(cur)
                cur = ""
            continue
        test = cur + ch
        if cur and font_obj.text_length(test, fontsize=fontsize) > max_width:
            lines.append(cur.rstrip())
            cur = ch.lstrip()
        else:
            cur = test
    if cur:
        lines.append(cur.rstrip())
    return lines


def _render_poetry_work_png(work: Any) -> bytes:
    """
    KoreanPoetryWorkAnnotation을 입력받아 명조체 시 원문 + 컬러 고딕 행간 주석 +
    밑줄·물결선·네모·동그라미·세모 + 좌·우 여백 브라켓/수미상관 호를 고해상도 PNG로 렌더링한다.
    """
    serif_path, sans_path, bold_path = _pick_korean_font_paths()
    serif_font = pymupdf.Font(fontfile=serif_path) if serif_path else pymupdf.Font("korea")
    sans_font = pymupdf.Font(fontfile=sans_path) if sans_path else pymupdf.Font("korea")
    bold_font = pymupdf.Font(fontfile=bold_path) if bold_path else sans_font

    color_map = {
        "red": (0.84, 0.16, 0.16),
        "blue": (0.11, 0.35, 0.78),
        "dark": (0.20, 0.25, 0.33),
    }
    poem_color = (0.08, 0.10, 0.15)
    muted_color = (0.38, 0.44, 0.54)

    raw_lines = [str(ln or "") for ln in (getattr(work, "lines", None) or [])]
    if not raw_lines:
        raise ValueError("시 원문 행 데이터가 비어 있습니다.")

    marks = list(getattr(work, "marks", None) or [])
    brackets = list(getattr(work, "brackets", None) or [])

    W = 520.0
    has_left_bracket = any((getattr(b, "side", "") or "") == "left" for b in brackets)
    X_LEFT_LABEL = 6.0
    X_LEFT_ARC = 64.0
    X_POEM = 72.0 if has_left_bracket else 22.0
    POEM_FS = 10.6
    NOTE_FS = 7.4
    MARGIN_FS = 7.5

    # 실제 렌더링 좌표로 전체 시 행 폭을 측정하고, 우측 여백(>=105pt)을 침범하면 폰트 크기 자동 축소
    measured_lines = [_measure_rendered_line_chars(ltxt, serif_path, POEM_FS) for ltxt in raw_lines]
    max_poem_w = max((lw for _, lw in measured_lines), default=120.0)
    max_allowed_poem_w = W - X_POEM - 115.0
    if max_poem_w > max_allowed_poem_w and max_poem_w > 0:
        POEM_FS = max(8.4, round(POEM_FS * (max_allowed_poem_w / max_poem_w), 2))
        measured_lines = [_measure_rendered_line_chars(ltxt, serif_path, POEM_FS) for ltxt in raw_lines]
        max_poem_w = max((lw for _, lw in measured_lines), default=120.0)

    X_RIGHT_MARGIN = max(350.0, X_POEM + max_poem_w + 18.0)
    if X_RIGHT_MARGIN > W - 88.0:
        X_RIGHT_MARGIN = W - 88.0
    RIGHT_TEXT_MAX_W = max(72.0, W - X_RIGHT_MARGIN - 8.0)

    # 각 mark를 실제 행 번호(1-based)에 정확히 매칭
    line_marks: dict[int, list[tuple[Any, int, int]]] = {}
    for mk in marks:
        q = (getattr(mk, "quote", "") or "").strip()
        if not q:
            continue
        li = int(getattr(mk, "line_index", 1) or 1)
        pos = -1
        target_li = li
        if 1 <= li <= len(raw_lines):
            pos = raw_lines[li - 1].find(q)
        if pos == -1:
            for cand_i, cand_txt in enumerate(raw_lines, start=1):
                p = cand_txt.find(q)
                if p != -1:
                    pos = p
                    target_li = cand_i
                    break
        if pos != -1:
            line_marks.setdefault(target_li, []).append((mk, pos, pos + len(q)))

    # 1차 패스: 각 행의 Y 좌표 및 행간 주석 배치(겹침 회피) 계산
    work_label = (getattr(work, "work_label", "") or "").strip()
    title_author = (getattr(work, "title_author", "") or "").strip()
    header_str = f"{work_label} {title_author}".strip() if work_label else title_author

    cur_y = 14.0
    if header_str:
        cur_y += 22.0

    line_layouts: list[dict[str, Any]] = []
    for li, ltxt in enumerate(raw_lines, start=1):
        if not ltxt.strip():
            line_layouts.append(
                {
                    "line_index": li,
                    "text": "",
                    "is_blank": True,
                    "y_base": cur_y + 5.0,
                    "y_mid": cur_y + 5.0,
                    "end_x": X_POEM,
                    "above_items": [],
                    "below_items": [],
                }
            )
            cur_y += 12.0
            continue

        chars_bbox, line_w = measured_lines[li - 1]
        m_list = sorted(line_marks.get(li, []), key=lambda t: t[1])
        above_items: list[dict[str, Any]] = []
        below_items: list[dict[str, Any]] = []

        for mk, p0, p1 in m_list:
            if chars_bbox and 0 <= p0 < p1 <= len(chars_bbox):
                x0 = X_POEM + chars_bbox[p0][0]
                x1 = X_POEM + chars_bbox[p1 - 1][1]
            else:
                x0 = X_POEM + serif_font.text_length(ltxt[:p0], fontsize=POEM_FS)
                x1 = x0 + serif_font.text_length(ltxt[p0:p1], fontsize=POEM_FS)
            mid_x = (x0 + x1) * 0.5
            note_txt = (getattr(mk, "note", "") or "").strip()
            note_w = sans_font.text_length(note_txt, fontsize=NOTE_FS)
            # 주석 텍스트 X 범위 [nx, nx + note_w]
            nx = max(X_POEM - 8.0, min(mid_x - note_w * 0.35, X_RIGHT_MARGIN - 8.0 - note_w))
            pos_pref = getattr(mk, "position", "below") or "below"

            item = {
                "mk": mk,
                "x0": x0,
                "x1": x1,
                "mid_x": mid_x,
                "nx": nx,
                "nw": note_w,
                "note": note_txt,
                "tier": 0,
            }

            def _overlaps(target_list: list[dict[str, Any]], cand_nx: float, cand_nw: float) -> bool:
                for ex in target_list:
                    if not (cand_nx + cand_nw + 6.0 < ex["nx"] or cand_nx > ex["nx"] + ex["nw"] + 6.0):
                        return True
                return False

            if pos_pref == "above":
                if not _overlaps(above_items, nx, note_w):
                    above_items.append(item)
                elif not _overlaps(below_items, nx, note_w):
                    below_items.append(item)
                else:
                    item["tier"] = 1
                    above_items.append(item)
            else:
                if not _overlaps(below_items, nx, note_w):
                    below_items.append(item)
                elif not _overlaps(above_items, nx, note_w):
                    above_items.append(item)
                else:
                    item["tier"] = 1
                    below_items.append(item)

        above_tiers = max((it["tier"] + 1 for it in above_items), default=0)
        below_tiers = max((it["tier"] + 1 for it in below_items), default=0)
        above_pad = 4.0 + above_tiers * 14.0
        below_pad = 7.0 + below_tiers * 15.5

        y_base = cur_y + above_pad + POEM_FS
        line_layouts.append(
            {
                "line_index": li,
                "text": ltxt,
                "is_blank": False,
                "y_base": y_base,
                "y_mid": y_base - POEM_FS * 0.35,
                "end_x": X_POEM + line_w,
                "above_items": above_items,
                "below_items": below_items,
            }
        )
        cur_y = y_base + below_pad

    H = max(120.0, cur_y + 16.0)

    # PyMuPDF 임시 문서 생성 및 폰트 등록
    doc = pymupdf.open()
    page = doc.new_page(width=W, height=H)
    if serif_path:
        page.insert_font(fontname="kserif", fontfile=serif_path)
        serif_fn = "kserif"
    else:
        serif_fn = "korea"
    if sans_path:
        page.insert_font(fontname="ksans", fontfile=sans_path)
        sans_fn = "ksans"
    else:
        sans_fn = "korea"
    if bold_path:
        page.insert_font(fontname="kbold", fontfile=bold_path)
        bold_fn = "kbold"
    else:
        bold_fn = sans_fn

    # 배경 카드 테두리
    page.draw_rect(pymupdf.Rect(1.5, 1.5, W - 1.5, H - 1.5), color=(0.82, 0.86, 0.92), fill=(0.99, 0.995, 1.0), width=0.8)

    # 작품 제목 헤더
    if header_str:
        page.insert_text((14.0, 24.0), header_str, fontname=bold_fn, fontsize=11.0, color=(0.08, 0.14, 0.28))
        page.draw_line((14.0, 29.0), (W - 14.0, 29.0), color=(0.85, 0.88, 0.93), width=0.7)

    # 본문 시 행 + 마크 + 행간 주석 렌더링
    for lay in line_layouts:
        if lay["is_blank"]:
            continue
        y_base = lay["y_base"]
        page.insert_text((X_POEM, y_base), lay["text"], fontname=serif_fn, fontsize=POEM_FS, color=poem_color)

        for pos_kind, items in (("above", lay["above_items"]), ("below", lay["below_items"])):
            for it in items:
                mk = it["mk"]
                col = color_map.get(getattr(mk, "color", "red"), color_map["red"])
                mtype = getattr(mk, "mark_type", "underline") or "underline"
                x0, x1, mid_x = it["x0"], it["x1"], it["mid_x"]
                y_top = y_base - POEM_FS * 0.85
                y_bot = y_base + 2.2

                if mtype == "underline":
                    page.draw_line((x0, y_bot), (x1, y_bot), color=col, width=1.1)
                    anchor_y = y_bot if pos_kind == "below" else y_top
                elif mtype == "wavy":
                    pts = []
                    wx = x0
                    step = 3.0
                    idx_w = 0
                    while wx <= x1:
                        wy = y_bot + (1.4 if idx_w % 2 == 0 else -1.0)
                        pts.append((wx, wy))
                        wx += step
                        idx_w += 1
                    if pts and pts[-1][0] < x1:
                        pts.append((x1, y_bot))
                    if len(pts) >= 2:
                        page.draw_polyline(pts, color=col, width=1.0)
                    anchor_y = y_bot + 1.5 if pos_kind == "below" else y_top
                elif mtype == "box":
                    page.draw_rect(pymupdf.Rect(x0 - 1.5, y_top - 1.0, x1 + 1.5, y_bot + 0.8), color=col, width=1.0)
                    anchor_y = y_bot + 0.8 if pos_kind == "below" else y_top - 1.0
                elif mtype == "circle":
                    page.draw_oval(pymupdf.Rect(x0 - 2.2, y_top - 1.5, x1 + 2.2, y_bot + 1.2), color=col, width=1.0)
                    anchor_y = y_bot + 1.2 if pos_kind == "below" else y_top - 1.5
                elif mtype == "triangle":
                    q_len = len((getattr(mk, "quote", "") or "").strip())
                    if q_len <= 2:
                        tri_pts = [(mid_x, y_top - 4.0), (x0 - 3.5, y_bot + 1.2), (x1 + 3.5, y_bot + 1.2)]
                        page.draw_polyline(tri_pts + [tri_pts[0]], color=col, width=1.05)
                        anchor_y = y_bot + 1.2 if pos_kind == "below" else y_top - 4.0
                    else:
                        page.draw_line((x0, y_bot), (x1, y_bot), color=col, width=1.1)
                        tri_cx = mid_x
                        tri_top = y_top - 4.2
                        tri_bot = y_top - 0.5
                        tri_pts = [(tri_cx, tri_top), (tri_cx - 2.6, tri_bot), (tri_cx + 2.6, tri_bot)]
                        page.draw_polyline(tri_pts + [tri_pts[0]], color=col, width=0.95)
                        anchor_y = y_bot if pos_kind == "below" else tri_top - 1.5
                else:
                    page.draw_line((x0, y_bot), (x1, y_bot), color=col, width=1.0)
                    anchor_y = y_bot

                if it["note"]:
                    tier = it["tier"]
                    if pos_kind == "below":
                        note_y = y_base + 14.5 + tier * 14.5
                        line_end_y = note_y - NOTE_FS - 1.5
                        page.draw_line((mid_x, anchor_y + 0.5), (mid_x, max(anchor_y + 2.0, line_end_y)), color=col, width=0.75)
                    else:
                        note_y = min(y_top - 4.5, anchor_y - 2.5) - tier * 14.0
                        line_end_y = note_y + 1.5
                        page.draw_line((mid_x, anchor_y - 0.5), (mid_x, min(anchor_y - 1.5, line_end_y)), color=col, width=0.75)
                    page.insert_text((it["nx"], note_y), it["note"], fontname=sans_fn, fontsize=NOTE_FS, color=col)

    # 좌·우 여백 브라켓 / 수미상관 호 / 시상전환 구분선 렌더링
    n_lays = len(line_layouts)
    for br in brackets:
        s_idx = max(1, min(int(getattr(br, "start_line", 1) or 1), n_lays))
        e_idx = max(s_idx, min(int(getattr(br, "end_line", s_idx) or s_idx), n_lays))
        y_s = line_layouts[s_idx - 1]["y_mid"] - 3.0
        y_e = line_layouts[e_idx - 1]["y_mid"] + 3.0
        y_mid = (y_s + y_e) * 0.5
        col = color_map.get(getattr(br, "color", "blue"), color_map["blue"])
        style = getattr(br, "style", "brace") or "brace"
        side = getattr(br, "side", "right") or "right"
        note_txt = (getattr(br, "note", "") or "").strip()

        if style == "divider":
            y_div = (line_layouts[s_idx - 1]["y_base"] + line_layouts[e_idx - 1]["y_mid"]) * 0.5
            page.draw_line((X_POEM - 8.0, y_div), (W - 14.0, y_div), color=col, width=0.9, dashes="[3 3] 0")
            if note_txt:
                wrapped = _wrap_text_by_width(note_txt, sans_font, MARGIN_FS, RIGHT_TEXT_MAX_W)
                for wi, wl in enumerate(wrapped[:3]):
                    page.insert_text((X_RIGHT_MARGIN, y_div - 3.0 + wi * 10.0), wl, fontname=bold_fn, fontsize=MARGIN_FS, color=col)
        elif style == "arc":
            ax = X_LEFT_ARC if side == "left" else X_RIGHT_MARGIN - 8.0
            ctrl_x = ax - 22.0 if side == "left" else ax + 22.0
            dy = y_e - y_s
            page.draw_bezier(
                (ax, y_s),
                (ctrl_x, y_s + dy * 0.25),
                (ctrl_x, y_s + dy * 0.75),
                (ax, y_e),
                color=col,
                width=1.1,
            )
            if note_txt:
                if side == "left":
                    wrapped = _wrap_text_by_width(note_txt, sans_font, 7.2, 42.0)
                    start_ty = y_mid - (len(wrapped) - 1) * 4.5
                    for wi, wl in enumerate(wrapped[:4]):
                        page.insert_text((X_LEFT_LABEL, start_ty + wi * 9.5), wl, fontname=bold_fn, fontsize=7.2, color=col)
                else:
                    wrapped = _wrap_text_by_width(note_txt, sans_font, MARGIN_FS, RIGHT_TEXT_MAX_W)
                    start_ty = y_mid - (len(wrapped) - 1) * 4.5
                    for wi, wl in enumerate(wrapped[:4]):
                        page.insert_text((X_RIGHT_MARGIN, start_ty + wi * 10.0), wl, fontname=bold_fn, fontsize=MARGIN_FS, color=col)
        else:
            # brace (중괄호 } / {) — 시 본문 끝(max_end_x)보다 항상 우측에 배치
            max_end_x = max((line_layouts[k]["end_x"] for k in range(s_idx - 1, e_idx)), default=X_POEM + 100.0)
            bx = min(W - 82.0, max_end_x + 8.0) if side == "right" else X_LEFT_ARC
            tip = bx + 5.0 if side == "right" else bx - 5.0
            pts = [(bx, y_s), (bx + 2.5 if side == "right" else bx - 2.5, y_s + 2.0), (bx + 2.5 if side == "right" else bx - 2.5, y_mid - 2.0), (tip, y_mid), (bx + 2.5 if side == "right" else bx - 2.5, y_mid + 2.0), (bx + 2.5 if side == "right" else bx - 2.5, y_e - 2.0), (bx, y_e)]
            page.draw_polyline(pts, color=col, width=1.0)
            if note_txt:
                tx = tip + 5.0 if side == "right" else X_LEFT_LABEL
                tw = (W - tx - 8.0) if side == "right" else 44.0
                wrapped = _wrap_text_by_width(note_txt, sans_font, MARGIN_FS, max(38.0, tw))
                start_ty = y_mid - (len(wrapped) - 1) * 4.5 + 2.0
                for wi, wl in enumerate(wrapped[:4]):
                    page.insert_text((tx, start_ty + wi * 10.0), wl, fontname=sans_fn, fontsize=MARGIN_FS, color=col)

    try:
        pix = page.get_pixmap(dpi=240, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def _render_poetry_work_png_pages(work: Any, max_height_pt: float = 560.0) -> list[bytes]:
    """시 원문 이미지가 너무 길어 페이지에 축소되지 않도록 연 단위로 분할해 렌더링한다."""
    from types import SimpleNamespace

    png0 = _render_poetry_work_png(work)
    with Image.open(io.BytesIO(png0)) as im:
        w_px, h_px = im.size
    h_pt = h_px * (520.0 / max(1, w_px))
    if h_pt <= max_height_pt:
        return [png0]

    raw_lines = [str(ln or "") for ln in (getattr(work, "lines", None) or [])]
    n_lines = len(raw_lines)
    if n_lines < 8:
        return [png0]

    n_chunks = max(2, math.ceil(h_pt / max_height_pt))
    blank_indices = [i for i, ln in enumerate(raw_lines) if not ln.strip()]

    split_points: list[int] = [0]
    for k in range(1, n_chunks):
        ideal = int(round(n_lines * k / n_chunks))
        # 가장 가까운 연 구분(빈 줄) 찾기
        best_bp = None
        best_dist = 9999
        for bi in blank_indices:
            if bi > split_points[-1] + 3 and bi < n_lines - 3:
                d = abs(bi - ideal)
                if d < best_dist:
                    best_dist = d
                    best_bp = bi
        if best_bp is not None and best_dist <= max(6, n_lines // (n_chunks * 2) + 2):
            split_points.append(best_bp + 1)
        elif ideal > split_points[-1] + 3 and ideal < n_lines - 2:
            split_points.append(ideal)
    split_points.append(n_lines)

    marks = list(getattr(work, "marks", None) or [])
    brackets = list(getattr(work, "brackets", None) or [])
    work_label = (getattr(work, "work_label", "") or "").strip()
    title_author = (getattr(work, "title_author", "") or "").strip()

    out_pngs: list[bytes] = []
    total_parts = len(split_points) - 1
    for part_idx in range(total_parts):
        s0 = split_points[part_idx]
        e0 = split_points[part_idx + 1]
        sub_lines = raw_lines[s0:e0]
        while sub_lines and not sub_lines[0].strip():
            sub_lines = sub_lines[1:]
            s0 += 1
        while sub_lines and not sub_lines[-1].strip():
            sub_lines = sub_lines[:-1]
            e0 -= 1
        if not sub_lines:
            continue

        # 1-based range [s0+1, e0]
        sub_marks = []
        for mk in marks:
            li = int(getattr(mk, "line_index", 0) or 0)
            if s0 + 1 <= li <= e0:
                sub_marks.append(
                    SimpleNamespace(
                        line_index=li - s0,
                        quote=getattr(mk, "quote", ""),
                        mark_type=getattr(mk, "mark_type", "underline"),
                        color=getattr(mk, "color", "red"),
                        position=getattr(mk, "position", "below"),
                        note=getattr(mk, "note", ""),
                    )
                )

        sub_brackets = []
        for br in brackets:
            bs = int(getattr(br, "start_line", 1) or 1)
            be = int(getattr(br, "end_line", bs) or bs)
            ov_s = max(bs, s0 + 1)
            ov_e = min(be, e0)
            if ov_s <= ov_e:
                sub_brackets.append(
                    SimpleNamespace(
                        start_line=ov_s - s0,
                        end_line=ov_e - s0,
                        side=getattr(br, "side", "right"),
                        style=getattr(br, "style", "brace"),
                        color=getattr(br, "color", "blue"),
                        note=getattr(br, "note", ""),
                    )
                )

        sub_title = f"{title_author} ({part_idx + 1}/{total_parts})" if total_parts > 1 else title_author
        sub_work = SimpleNamespace(
            work_label=work_label if part_idx == 0 else "",
            title_author=sub_title,
            lines=sub_lines,
            marks=sub_marks,
            brackets=sub_brackets,
        )
        out_pngs.append(_render_poetry_work_png(sub_work))

    return out_pngs or [png0]


def _highlight_poem_line_html(
    line_text: str,
    notes_with_idx: list[tuple[int, str]],
    matched_indices: set[int],
) -> str:
    """원문 시 행에서 아직 매칭되지 않은 시구(quote)를 찾아 밑줄 + 원문자 번호 HTML로 변환한다."""
    segments: list[tuple[bool, str, int]] = [(False, line_text, -1)]
    for note_idx, quote in notes_with_idx:
        if note_idx in matched_indices or not quote:
            continue
        new_segments: list[tuple[bool, str, int]] = []
        found_here = False
        for is_match, seg_txt, seg_idx in segments:
            if is_match or found_here:
                new_segments.append((is_match, seg_txt, seg_idx))
                continue
            pos = seg_txt.find(quote)
            if pos != -1:
                if pos > 0:
                    new_segments.append((False, seg_txt[:pos], -1))
                new_segments.append((True, seg_txt[pos : pos + len(quote)], note_idx))
                if pos + len(quote) < len(seg_txt):
                    new_segments.append((False, seg_txt[pos + len(quote) :], -1))
                matched_indices.add(note_idx)
                found_here = True
            else:
                new_segments.append((False, seg_txt, -1))
        segments = new_segments

    out_parts: list[str] = []
    for is_match, seg_txt, seg_idx in segments:
        if is_match:
            badge = _CIRCLED_NUMS[seg_idx] if seg_idx < len(_CIRCLED_NUMS) else f"({seg_idx + 1})"
            out_parts.append(
                f"<span class='poem-ul'><u>{html.escape(seg_txt)}</u></span>"
                f"<span class='poem-badge'> {badge}</span>"
            )
        else:
            out_parts.append(html.escape(seg_txt))
    return "".join(out_parts)
