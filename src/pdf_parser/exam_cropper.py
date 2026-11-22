from __future__ import annotations

import io
import re

import pymupdf
from PIL import Image

from src.pdf_parser.common import (
    QUESTION_START_RE,
    SHARED_PASSAGE_RE,
    CroppedQuestion,
)

def _detect_columns(page: pymupdf.Page) -> list[tuple[float, float]]:
    """
    모의고사 PDF의 가로 중앙(page_width / 2)을 기준으로 왼쪽 단 -> 오른쪽 단 영역을 반환한다.
    명백한 1단 통짜 본문(구분선 없고 중앙을 가로지르는 긴 문단 다수)만 예외적으로 1단 처리한다.
    """
    w = page.rect.width
    h = page.rect.height
    mid_x = w / 2.0

    has_center_divider = False
    for d in page.get_drawings():
        rect = d.get("rect")
        if not rect:
            continue
        rw = abs(rect.x1 - rect.x0)
        rh = abs(rect.y1 - rect.y0)
        cx = (rect.x0 + rect.x1) / 2.0
        if (0.44 * w <= cx <= 0.56 * w) and rw <= 5.0 and rh >= 0.20 * h:
            has_center_divider = True
            break

    if has_center_divider:
        return [(0.0, mid_x), (mid_x, w)]

    # 중앙을 가로지르는 순수 1단 문단이 다수이고 오른쪽 단독 컬럼이 전혀 없는 경우에만 1단으로 처리
    spanning_lines = 0
    right_only_lines = 0
    page_dict = page.get_text("dict")
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            lx0, ly0, lx1, ly1 = line["bbox"]
            if ly1 < 0.14 * h or ly0 > 0.92 * h:
                continue
            txt = "".join(sp.get("text", "") for sp in line.get("spans", [])).strip()
            if len(txt) < 8:
                continue
            if lx0 < 0.30 * w and lx1 > 0.70 * w:
                spanning_lines += 1
            elif lx0 >= mid_x - 4.0:
                right_only_lines += 1

    if spanning_lines >= 4 and right_only_lines == 0:
        return [(0.0, w)]
    return [(0.0, mid_x), (mid_x, w)]


_PDF_HYPHEN_CHARS_RE = re.compile(r"[\u00ad\u2010\u2011\u2012\u2212\uf02d]")
_PDF_ENDASH_ALNUM_RE = re.compile(r"(?<=[A-Za-z0-9])\u2013(?=[A-Za-z0-9])")


def _normalize_pdf_hyphens(text: str) -> str:
    """PDF 추출 텍스트의 소프트 하이픈·PUA 하이픈·유니코드 대시를 ASCII '-'로 정규화한다."""
    if not text:
        return ""
    out = _PDF_HYPHEN_CHARS_RE.sub("-", text)
    out = _PDF_ENDASH_ALNUM_RE.sub("-", out)
    return out


def _has_vector_hyphen_between(
    x_gap_start: float,
    x_gap_end: float,
    line_y0: float,
    line_y1: float,
    drawings: list[dict] | None,
) -> bool:
    """두 span 사이 간격 안에 벡터 선분으로 그려진 짧은 수평 하이픈이 있는지 판별한다."""
    if not drawings:
        return False
    line_h = max(4.0, line_y1 - line_y0)
    y_lo = line_y0 + 0.15 * line_h
    y_hi = line_y1 - 0.15 * line_h
    x_lo = x_gap_start - 1.5
    x_hi = x_gap_end + 1.5
    for d in drawings:
        rect = d.get("rect")
        if not rect:
            continue
        rw = float(rect.x1 - rect.x0)
        rh = float(rect.y1 - rect.y0)
        if 1.5 <= rw <= 11.0 and rh <= 2.2:
            rcx = (rect.x0 + rect.x1) / 2.0
            rcy = (rect.y0 + rect.y1) / 2.0
            if x_lo <= rcx <= x_hi and y_lo <= rcy <= y_hi:
                return True
    return False


def _extract_lines_in_column(
    page_dict: dict,
    col_x0: float,
    col_x1: float,
    page_h: float,
    page_w: float,
    page_drawings: list[dict] | None = None,
) -> list[dict]:
    """
    블록 경계가 2단을 가로지르더라도 span 단위로 칼럼 소속을 분리해
    해당 단 내 텍스트 라인만 y좌표 -> x좌표 순으로 정렬해 반환한다.
    """
    is_two_col = (col_x1 - col_x0) < (0.85 * page_w)
    lines: list[dict] = []

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            lx0, ly0, lx1, ly1 = line["bbox"]
            if ly1 < 0.035 * page_h or ly0 > 0.945 * page_h:
                continue

            col_spans: list[dict] = []
            for sp in line.get("spans", []):
                stext = sp.get("text", "")
                if not stext:
                    continue
                sx0, sy0, sx1, sy1 = sp.get("bbox", (lx0, ly0, lx1, ly1))
                scx = (sx0 + sx1) / 2.0
                if is_two_col:
                    if col_x0 == 0.0:
                        if not (scx < col_x1):
                            continue
                    else:
                        if not (col_x0 <= scx <= col_x1):
                            continue
                col_spans.append(sp)

            if not col_spans:
                continue

            parts: list[str] = []
            for idx_sp, sp in enumerate(col_spans):
                st = _normalize_pdf_hyphens(sp.get("text", ""))
                if idx_sp > 0 and parts:
                    prev_sp = col_spans[idx_sp - 1]
                    prev_txt = parts[-1]
                    prev_bbox = prev_sp.get("bbox", (lx0, ly0, lx1, ly1))
                    curr_bbox = sp.get("bbox", (lx0, ly0, lx1, ly1))
                    gap = float(curr_bbox[0]) - float(prev_bbox[2])
                    if (
                        0.5 <= gap <= 14.0
                        and prev_txt
                        and st
                        and prev_txt[-1].isalnum()
                        and st[0].isalnum()
                        and _has_vector_hyphen_between(
                            float(prev_bbox[2]),
                            float(curr_bbox[0]),
                            float(ly0),
                            float(ly1),
                            page_drawings,
                        )
                    ):
                        parts.append("-")
                parts.append(st)

            text = "".join(parts).strip()
            if not text:
                continue

            sx0 = min(sp.get("bbox", (lx0, ly0, lx1, ly1))[0] for sp in col_spans)
            sy0 = min(sp.get("bbox", (lx0, ly0, lx1, ly1))[1] for sp in col_spans)
            sx1 = max(sp.get("bbox", (lx0, ly0, lx1, ly1))[2] for sp in col_spans)
            sy1 = max(sp.get("bbox", (lx0, ly0, lx1, ly1))[3] for sp in col_spans)
            max_font_size = max((sp.get("size", 0.0) for sp in col_spans), default=0.0)
            is_bold = any(bool(sp.get("flags", 0) & 16) for sp in col_spans)
            lines.append(
                {
                    "x0": sx0,
                    "y0": sy0,
                    "x1": sx1,
                    "y1": sy1,
                    "text": text,
                    "font_size": max_font_size,
                    "is_bold": is_bold,
                }
            )

    lines.sort(key=lambda item: (round(item["y0"], 1), item["x0"]))
    return lines


def _is_footer_or_header_noise(text: str, y0: float, page_h: float) -> bool:
    # 본문 중앙에서 우연히 단어가 겹치는 오탐을 막기 위해 상·하단 여백 구간 위주로 필터링
    t = (text or "").strip()
    in_top_margin = y0 < 0.095 * page_h
    in_bottom_margin = y0 > 0.895 * page_h
    in_margin = in_top_margin or in_bottom_margin

    always_noise = (
        "이 문제지에 관한 저작권",
        "한국교육과정평가원",
        "답안지의 해당란에",
    )
    if any(p in t for p in always_noise):
        return True
    if re.match(r"^(?:국어|수학|영어|한국사|사회탐구|과학탐구|직업탐구|제2외국어)\s*영역$", t):
        return True
    # 단독 1~2자리 쪽번호('7', '[10]' 등)는 약간 더 넓은 상·하단 여백까지 필터링 (마침표 붙은 문항번호 '39.'는 제외)
    if (y0 < 0.115 * page_h or y0 > 0.885 * page_h) and re.match(r"^\[?\s*\d{1,2}\s*\]?$", t):
        return True
    if not in_margin:
        return False

    # 상·하단 여백의 단독 쪽수('[10]', '12', '6') 또는 '12 국어', '국어 11', '영역 7' 등
    if re.match(r"^\[?\s*\d{1,2}\s*\]?$", t):
        return True
    if re.match(r"^(?:\d{1,2}\s+)?(?:국어|수학|영어|한국사|사회탐구|과학탐구|직업탐구|제2외국어)(?:\s+\d{1,2})?$", t):
        return True
    if re.match(r"^(?:영역\s*\d{1,2}|\d{1,2}\s*영역)$", t):
        return True

    margin_noise = (
        "학년도 대학수학능력시험",
        "전국연합학력평가",
        "모의평가 문제지",
        "확인 사항",
        "제 1 교시",
        "제 2 교시",
        "제 3 교시",
        "제 4 교시",
        "영역",
    )
    return any(p in t for p in margin_noise)


def _collect_content_boxes_in_column(
    page: pymupdf.Page,
    page_dict: dict,
    col_x0: float,
    col_x1: float,
    page_h: float,
) -> list[tuple[float, float, float, float]]:
    """정밀 크롭 경계 계산을 위해 컬럼 내 텍스트/이미지/도형 박스를 수집한다 (상·하단 헤더/푸터 제외)."""
    boxes: list[tuple[float, float, float, float]] = []

    for block in page_dict.get("blocks", []):
        btype = block.get("type")
        bx0, by0, bx1, by1 = block["bbox"]
        if by1 < 0.055 * page_h or by0 > 0.905 * page_h:
            continue

        if btype == 0:
            for ln in block.get("lines", []):
                lx0, ly0, lx1, ly1 = ln["bbox"]
                if ly1 < 0.055 * page_h or ly0 > 0.905 * page_h:
                    continue
                ln_text = "".join(sp.get("text", "") for sp in ln.get("spans", [])).strip()
                if not ln_text or _is_footer_or_header_noise(ln_text, ly0, page_h):
                    continue
                for sp in ln.get("spans", []):
                    if not sp.get("text", "").strip():
                        continue
                    sx0, sy0, sx1, sy1 = sp.get("bbox", (lx0, ly0, lx1, ly1))
                    scx = (sx0 + sx1) / 2.0
                    if col_x0 <= scx <= col_x1:
                        boxes.append((max(col_x0, sx0), sy0, min(col_x1, sx1), sy1))
        else:
            bcx = (bx0 + bx1) / 2.0
            overlap_x = min(col_x1, bx1) - max(col_x0, bx0)
            if (col_x0 <= bcx <= col_x1) or overlap_x > 16.0:
                boxes.append((max(col_x0, bx0), by0, min(col_x1, bx1), by1))

    for d in page.get_drawings():
        rect = d.get("rect")
        if not rect:
            continue
        rw = rect.x1 - rect.x0
        rh = rect.y1 - rect.y0
        if rh > 0.72 * page_h and rw < 6.0:
            continue
        if rw > 0.85 * page.rect.width:
            continue
        # 상·하단 헤더/푸터 가로 구분선 및 페이지 번호 박스 제외
        if rect.y1 < 0.095 * page_h and rh < 6.0 and rw > 0.25 * (col_x1 - col_x0):
            continue
        dcx = (rect.x0 + rect.x1) / 2.0
        if not (col_x0 <= dcx <= col_x1):
            continue
        if rect.y1 < 0.055 * page_h or rect.y0 > 0.90 * page_h:
            continue
        boxes.append((max(col_x0, rect.x0), rect.y0, min(col_x1, rect.x1), rect.y1))

    return boxes


def _stitch_vertical_pngs(top_png: bytes, bottom_png: bytes, gap_px: int = 12) -> tuple[bytes, int, int]:
    """이전 단(컬럼)의 세트 지문 PNG와 문항 PNG를 위아래로 깔끔하게 이어 붙인다."""
    with Image.open(io.BytesIO(top_png)) as im_top, Image.open(io.BytesIO(bottom_png)) as im_bot:
        w = max(im_top.width, im_bot.width)
        h = im_top.height + gap_px + im_bot.height
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        canvas.paste(im_top.convert("RGB"), (0, 0))
        canvas.paste(im_bot.convert("RGB"), (0, im_top.height + gap_px))
        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        return buf.getvalue(), w, h


def parse_exam_pdf(
    pdf_bytes: bytes,
    dpi: int = 250,
    split_passage_and_question: bool = False,
) -> tuple[list[CroppedQuestion], bool]:
    """
    2단 편집 한국어 모의고사/수능 문제지 PDF를 PyMuPDF로 분석해
    가로 중앙(page_width / 2) 기준 왼쪽 단(위->아래) 후 오른쪽 단(위->아래) 순서로
    1번~45번 전체 문항 및 세트 지문([N~M])을 고해상도(>=220 DPI) PNG로 크롭한다.
    split_passage_and_question=True(국어 등)일 때는 공통 지문 이미지와 개별 문제 이미지를 분리해 반환한다.
    """
    effective_dpi = max(220, dpi)
    results: list[CroppedQuestion] = []
    seen_q_nums: set[int] = set()
    total_text_chars = 0

    # 이전 단(또는 이전 페이지) 끝에서 미처 첫 문항을 만나지 못한 세트 지문 [N~M] 이월용 및 세트 문항 공통 텍스트 유지용
    carried_passage: dict | None = None
    active_passage_group: dict | None = None
    # 이전 단 맨 아래 문항이 다음 단 상단(새 문항/새 지문 시작 전)까지 이어지는 경우(선지 등) 꼬리 병합용
    prev_col_last_q_idx: int | None = None

    def _crop_region_png(
        page_obj: pymupdf.Page,
        c_boxes: list[tuple[float, float, float, float]],
        c_x0: float,
        c_x1: float,
        p_h: float,
        y_lo: float,
        y_hi: float,
    ) -> tuple[bytes, int, int] | None:
        r_boxes = [b for b in c_boxes if b[3] > y_lo - 4.0 and b[1] < y_hi]
        if not r_boxes:
            return None
        r_clip = pymupdf.Rect(
            max(c_x0 + 2.0, min(b[0] for b in r_boxes) - 10.0),
            max(0.055 * p_h, min(y_lo, min(b[1] for b in r_boxes)) - 6.0),
            min(c_x1 - 2.0, max(b[2] for b in r_boxes) + 10.0),
            min(y_hi, max(b[3] for b in r_boxes) + 8.0),
        )
        if r_clip.y1 <= r_clip.y0 + 6.0:
            return None
        r_pix = page_obj.get_pixmap(dpi=effective_dpi, clip=r_clip, alpha=False)
        return r_pix.tobytes("png"), r_pix.width, r_pix.height

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_idx in range(doc.page_count):
            page = doc[page_idx]
            page_h = page.rect.height
            page_w = page.rect.width
            page_dict = page.get_text("dict")
            page_drawings = page.get_drawings()
            total_text_chars += len(re.sub(r"\s+", "", page.get_text("text") or ""))

            columns = _detect_columns(page)

            for col_x0, col_x1 in columns:
                col_w = col_x1 - col_x0
                lines = _extract_lines_in_column(
                    page_dict,
                    col_x0,
                    col_x1,
                    page_h,
                    page_w,
                    page_drawings=page_drawings,
                )
                if not lines:
                    continue

                # 상단 여백 헤더('16 국어', '12 국어' 등)가 있으면 그 아래부터 본문 시작점으로 제한
                top_header_y1 = 0.055 * page_h
                for ln in lines:
                    if ln["y0"] < 0.105 * page_h and _is_footer_or_header_noise(ln["text"], ln["y0"], page_h):
                        top_header_y1 = max(top_header_y1, ln["y1"] + 3.0)

                # 컬럼 본문의 좌측 시작 기준 (과도한 좌측 여백 노이즈 방지를 위해 중앙값/대다수 기준 보조)
                candidate_xs = sorted(
                    ln["x0"]
                    for ln in lines
                    if not _is_footer_or_header_noise(ln["text"], ln["y0"], page_h)
                    and QUESTION_START_RE.match(ln["text"])
                )
                ref_q_x0 = candidate_xs[0] if candidate_xs else (col_x0 + 24.0)
                max_allowed_x0 = max(col_x0 + 0.34 * col_w, ref_q_x0 + 48.0)

                markers: list[dict] = []
                first_passage_y0_in_col: float | None = None
                pending_passage_y0: float | None = None
                pending_passage_start_q: int | None = None
                pending_passage_end_q: int | None = None
                same_col_passage_crops: list[tuple[str, float, float]] = []

                for ln in lines:
                    txt = ln["text"]
                    if _is_footer_or_header_noise(txt, ln["y0"], page_h):
                        continue

                    # 공통/세트 지문 헤더 [1~3], [16~17], [41~42], [43~45] 등 감지
                    pm = SHARED_PASSAGE_RE.match(txt)
                    if pm and ln["x0"] <= col_x0 + 0.45 * col_w:
                        p_start = int(pm.group(1))
                        p_end = int(pm.group(2))
                        if 1 <= p_start <= p_end <= 45:
                            if first_passage_y0_in_col is None:
                                first_passage_y0_in_col = ln["y0"]
                            pending_passage_y0 = ln["y0"]
                            pending_passage_start_q = p_start
                            pending_passage_end_q = p_end
                        continue

                    qm = QUESTION_START_RE.match(txt)
                    if not qm:
                        continue
                    q_num = int(qm.group(1))
                    if not (1 <= q_num <= 45):
                        continue

                    if ln["x0"] > max_allowed_x0:
                        continue

                    # 좌->우, 위->아래 순차 탐색 중 중복 번호 및 역순 오탐(<보기> 내부 숫자 등) 방지
                    if q_num in seen_q_nums:
                        continue
                    if seen_q_nums and q_num < max(seen_q_nums):
                        continue

                    start_y = ln["y0"]
                    if (
                        pending_passage_y0 is not None
                        and pending_passage_start_q is not None
                        and pending_passage_end_q is not None
                        and q_num <= pending_passage_end_q
                    ):
                        same_col_shared_txt = "\n".join(
                            l_item["text"]
                            for l_item in lines
                            if pending_passage_y0 - 2.0 <= l_item["y0"] < ln["y0"] - 1.0
                            and not _is_footer_or_header_noise(l_item["text"], l_item["y0"], page_h)
                        ).strip()
                        grp_id = f"{pending_passage_start_q}~{pending_passage_end_q}"
                        active_passage_group = {
                            "start_q": pending_passage_start_q,
                            "end_q": pending_passage_end_q,
                            "group_id": grp_id,
                            "text": same_col_shared_txt,
                            "png_bytes": b"",
                            "width": 0,
                            "height": 0,
                        }
                        if split_passage_and_question:
                            # 지문과 문제를 분리할 때는 지문 영역을 별도 기록하고 문항은 번호 시작점(ln["y0"])부터 크롭
                            same_col_passage_crops.append((grp_id, pending_passage_y0, ln["y0"] - 3.0))
                        else:
                            start_y = min(start_y, pending_passage_y0)
                        pending_passage_y0 = None
                        pending_passage_start_q = None
                        pending_passage_end_q = None

                    markers.append(
                        {
                            "q_num": q_num,
                            "y_start": start_y,
                            "anchor_y": ln["y0"],
                            "passage_group": (
                                dict(active_passage_group)
                                if (
                                    active_passage_group is not None
                                    and active_passage_group["start_q"] <= q_num <= active_passage_group["end_q"]
                                )
                                else None
                            ),
                        }
                    )
                    seen_q_nums.add(q_num)

                content_boxes = [
                    b
                    for b in _collect_content_boxes_in_column(page, page_dict, col_x0, col_x1, page_h)
                    if b[3] > top_header_y1
                ]

                # 같은 단에 있던 지문 영역을 별도 PNG로 크롭(split_passage_and_question=True 전용)
                if split_passage_and_question and same_col_passage_crops:
                    for s_gid, py0, py1 in same_col_passage_crops:
                        cropped_p = _crop_region_png(page, content_boxes, col_x0, col_x1, page_h, py0, py1)
                        if cropped_p:
                            p_b, p_w, p_h_px = cropped_p
                            if active_passage_group and active_passage_group.get("group_id") == s_gid:
                                active_passage_group["png_bytes"] = p_b
                                active_passage_group["width"] = p_w
                                active_passage_group["height"] = p_h_px
                            for m_item in markers:
                                mg = m_item.get("passage_group")
                                if mg and mg.get("group_id") == s_gid:
                                    mg["png_bytes"] = p_b
                                    mg["width"] = p_w
                                    mg["height"] = p_h_px

                # 직전 단에서 이월된 세트 지문(예: [43~45] (A))이 현재 단 상단에서 (B)(C)(D) 등으로 이어진 뒤 첫 문항(43.)이 나오는 경우
                if (
                    markers
                    and carried_passage is not None
                    and carried_passage["start_q"] <= markers[0]["q_num"] <= carried_passage["end_q"]
                ):
                    first_m = markers[0]
                    cont_lines = [
                        ln
                        for ln in lines
                        if top_header_y1 <= ln["y0"] < first_m["y_start"] - 1.0
                        and not _is_footer_or_header_noise(ln["text"], ln["y0"], page_h)
                    ]
                    if cont_lines:
                        cont_txt = "\n".join(ln["text"] for ln in cont_lines).strip()
                        if cont_txt:
                            carried_passage["text"] = f"{carried_passage.get('text', '')}\n{cont_txt}".strip()
                            if (
                                active_passage_group is not None
                                and active_passage_group["start_q"] == carried_passage["start_q"]
                            ):
                                active_passage_group["text"] = carried_passage["text"]
                        if split_passage_and_question:
                            # 이어지는 지문 이미지를 carried_passage 지문 이미지 아래에 이어 붙임 (문제 이미지에는 안 붙임)
                            cont_crop = _crop_region_png(
                                page,
                                content_boxes,
                                col_x0,
                                col_x1,
                                page_h,
                                min(ln["y0"] for ln in cont_lines),
                                first_m["y_start"] - 3.0,
                            )
                            if cont_crop:
                                st_p, st_pw, st_ph = _stitch_vertical_pngs(
                                    carried_passage["png_bytes"], cont_crop[0]
                                )
                                carried_passage["png_bytes"] = st_p
                                carried_passage["width"] = st_pw
                                carried_passage["height"] = st_ph
                                if active_passage_group and active_passage_group["start_q"] == carried_passage["start_q"]:
                                    active_passage_group["png_bytes"] = st_p
                                    active_passage_group["width"] = st_pw
                                    active_passage_group["height"] = st_ph
                        else:
                            first_m["y_start"] = min(first_m["y_start"], min(ln["y0"] for ln in cont_lines))
                    if (
                        active_passage_group is not None
                        and active_passage_group["start_q"] <= first_m["q_num"] <= active_passage_group["end_q"]
                    ):
                        for m_item in markers:
                            if active_passage_group["start_q"] <= m_item["q_num"] <= active_passage_group["end_q"]:
                                m_item["passage_group"] = dict(active_passage_group)
                    prev_col_last_q_idx = None
                elif (
                    not markers
                    and carried_passage is not None
                    and pending_passage_y0 is None
                ):
                    # 한 단 전체가 이어지는 지문으로만 채워져 있고 문항이 하나도 없는 경우 (긴 국어 지문 대응)
                    mid_top_y = first_passage_y0_in_col if first_passage_y0_in_col is not None else 0.902 * page_h
                    mid_lines = [
                        ln
                        for ln in lines
                        if top_header_y1 <= ln["y0"] < mid_top_y - 1.0
                        and not _is_footer_or_header_noise(ln["text"], ln["y0"], page_h)
                    ]
                    if mid_lines:
                        mid_txt = "\n".join(ln["text"] for ln in mid_lines).strip()
                        if mid_txt:
                            carried_passage["text"] = f"{carried_passage.get('text', '')}\n{mid_txt}".strip()
                        mid_crop = _crop_region_png(
                            page, content_boxes, col_x0, col_x1, page_h, min(ln["y0"] for ln in mid_lines), mid_top_y
                        )
                        if mid_crop:
                            st_p, st_pw, st_ph = _stitch_vertical_pngs(
                                carried_passage["png_bytes"], mid_crop[0]
                            )
                            carried_passage["png_bytes"] = st_p
                            carried_passage["width"] = st_pw
                            carried_passage["height"] = st_ph
                            if active_passage_group and active_passage_group["start_q"] == carried_passage["start_q"]:
                                active_passage_group["text"] = carried_passage["text"]
                                active_passage_group["png_bytes"] = st_p
                                active_passage_group["width"] = st_pw
                                active_passage_group["height"] = st_ph
                    prev_col_last_q_idx = None
                elif prev_col_last_q_idx is not None:
                    # 직전 단의 마지막 문항(예: 17번)이 현재 단 상단(첫 문항 또는 첫 세트 헤더 이전)까지 이어지는 경우 꼬리(①~⑤ 선지 등) 병합
                    # 단, 새 지문('다음 글을 읽고 물음에 답하시오', '[N~M]')이 시작되는 경우는 절대 병합하지 않음
                    boundary_candidates: list[float] = []
                    if markers:
                        boundary_candidates.append(markers[0]["y_start"])
                    if first_passage_y0_in_col is not None:
                        boundary_candidates.append(first_passage_y0_in_col)
                    top_boundary_y = min(boundary_candidates) if boundary_candidates else (0.905 * page_h)

                    tail_lines = [
                        ln
                        for ln in lines
                        if top_header_y1 <= ln["y0"] < top_boundary_y - 1.0
                        and not _is_footer_or_header_noise(ln["text"], ln["y0"], page_h)
                    ]
                    tail_head_blob = " ".join(ln["text"] for ln in tail_lines[:3])
                    is_new_passage_start = (
                        "다음 글을 읽고 물음에 답하시오" in tail_head_blob
                        or any(SHARED_PASSAGE_RE.match(ln["text"]) for ln in tail_lines[:3])
                    )
                    if tail_lines and not is_new_passage_start and len(tail_lines) <= 18:
                        tail_top = min(ln["y0"] for ln in tail_lines) - 4.0
                        tail_boxes = [
                            b
                            for b in content_boxes
                            if b[3] > tail_top and b[1] < top_boundary_y - 4.0
                        ]
                        if tail_boxes:
                            t_clip = pymupdf.Rect(
                                max(col_x0 + 2.0, min(b[0] for b in tail_boxes) - 10.0),
                                max(top_header_y1, min(tail_top, min(b[1] for b in tail_boxes)) - 6.0),
                                min(col_x1 - 2.0, max(b[2] for b in tail_boxes) + 10.0),
                                min(top_boundary_y - 3.0, max(b[3] for b in tail_boxes) + 8.0),
                            )
                            if 8.0 < (t_clip.y1 - t_clip.y0) <= 0.45 * page_h:
                                t_pix = page.get_pixmap(dpi=effective_dpi, clip=t_clip, alpha=False)
                                prev_cq = results[prev_col_last_q_idx]
                                stitched_png, st_w, st_h = _stitch_vertical_pngs(
                                    prev_cq.png_bytes, t_pix.tobytes("png")
                                )
                                prev_cq.png_bytes = stitched_png
                                prev_cq.width = st_w
                                prev_cq.height = st_h
                                tail_txt = "\n".join(ln["text"] for ln in tail_lines).strip()
                                if tail_txt:
                                    prev_cq.extracted_text = f"{prev_cq.extracted_text}\n{tail_txt}".strip()
                    prev_col_last_q_idx = None

                for i, marker in enumerate(markers):
                    is_last_in_col = i == len(markers) - 1
                    trailing_passage_cuts_col = (
                        is_last_in_col
                        and pending_passage_y0 is not None
                        and pending_passage_y0 > marker["y_start"]
                    )
                    y_top_limit = marker["y_start"] - 4.0
                    if not is_last_in_col:
                        y_bottom_limit = markers[i + 1]["y_start"] - 4.0
                    elif trailing_passage_cuts_col:
                        y_bottom_limit = pending_passage_y0 - 6.0
                    else:
                        # 각 단 맨 아래 문항은 하단 푸터('12 국어', '영역 7', 쪽번호) 위쪽까지만 확보
                        y_bottom_limit = 0.905 * page_h

                    matched_boxes = [
                        b
                        for b in content_boxes
                        if b[3] > y_top_limit and b[1] < y_bottom_limit and b[1] >= y_top_limit - 8.0
                    ]

                    if matched_boxes:
                        min_x = min(b[0] for b in matched_boxes)
                        min_y = min(marker["y_start"], min(b[1] for b in matched_boxes))
                        max_x = max(b[2] for b in matched_boxes)
                        max_y = max(b[3] for b in matched_boxes)
                    else:
                        min_x = max(col_x0, col_x0 + 18.0)
                        min_y = marker["y_start"]
                        max_x = col_x1 - 8.0
                        max_y = y_bottom_limit

                    if max_y - min_y < 40.0:
                        max_y = min(y_bottom_limit, min_y + 120.0)

                    pad_x = 12.0
                    pad_y = 8.0
                    clip = pymupdf.Rect(
                        max(col_x0 + 2.0, min_x - pad_x),
                        max(top_header_y1, min_y - pad_y),
                        min(col_x1 - 2.0, max_x + pad_x),
                        min(
                            0.908 * page_h if (is_last_in_col and not trailing_passage_cuts_col) else y_bottom_limit,
                            max_y + pad_y,
                        ),
                    )
                    if clip.y1 <= clip.y0:
                        clip.y1 = min(0.908 * page_h, clip.y0 + 120.0)

                    if clip.width < 0.30 * page_w:
                        clip.x0 = max(col_x0 + 4.0, col_x0 + 12.0)
                        clip.x1 = min(col_x1 - 4.0, max(clip.x1, col_x1 - 16.0))

                    pix = page.get_pixmap(dpi=effective_dpi, clip=clip, alpha=False)
                    png_out = pix.tobytes("png")
                    out_w, out_h = pix.width, pix.height
                    q_text = "\n".join(
                        ln["text"]
                        for ln in lines
                        if ln["y1"] >= clip.y0 and ln["y0"] <= clip.y1
                        and not _is_footer_or_header_noise(ln["text"], ln["y0"], page_h)
                    )

                    q_passage_png = b""
                    q_passage_w = 0
                    q_passage_h = 0

                    # 직전 단에서 이월된 세트 지문이 현재 문항 번호 구간(start_q <= q_num <= end_q)에 속할 때
                    if (
                        carried_passage is not None
                        and carried_passage["start_q"] <= marker["q_num"] <= carried_passage["end_q"]
                    ):
                        if split_passage_and_question:
                            q_passage_png = carried_passage.get("png_bytes", b"")
                            with Image.open(io.BytesIO(q_passage_png)) as im_p:
                                q_passage_w, q_passage_h = im_p.size
                        else:
                            png_out, out_w, out_h = _stitch_vertical_pngs(
                                carried_passage["png_bytes"],
                                png_out,
                            )
                        if carried_passage.get("text"):
                            q_text = f"{carried_passage['text']}\n\n{q_text}".strip()
                        carried_passage = None
                    elif carried_passage is not None and marker["q_num"] > carried_passage["end_q"]:
                        carried_passage = None

                    q_group_id = ""
                    q_shared_text = ""
                    m_group = marker.get("passage_group")
                    if m_group is None and active_passage_group is not None:
                        if active_passage_group["start_q"] <= marker["q_num"] <= active_passage_group["end_q"]:
                            m_group = active_passage_group
                        elif marker["q_num"] > active_passage_group["end_q"]:
                            active_passage_group = None
                    if m_group is not None:
                        q_group_id = str(m_group.get("group_id") or "")
                        q_shared_text = str(m_group.get("text") or "").strip()
                        if q_shared_text and q_shared_text not in q_text:
                            q_text = f"{q_shared_text}\n\n{q_text}".strip()
                        if split_passage_and_question and not q_passage_png and m_group.get("png_bytes"):
                            q_passage_png = m_group["png_bytes"]
                            q_passage_w = int(m_group.get("width") or 0)
                            q_passage_h = int(m_group.get("height") or 0)

                    results.append(
                        CroppedQuestion(
                            q_num=marker["q_num"],
                            page_num=page_idx + 1,
                            png_bytes=png_out,
                            width=out_w,
                            height=out_h,
                            extracted_text=q_text,
                            passage_group_id=q_group_id,
                            shared_passage_text=q_shared_text,
                            passage_png_bytes=q_passage_png,
                            passage_width=q_passage_w,
                            passage_height=q_passage_h,
                        )
                    )
                    if is_last_in_col and not trailing_passage_cuts_col:
                        # 이미 ⑤(또는 5)) 선지까지 완결된 문항은 다음 단으로 꼬리를 이어 붙이지 않음
                        q_has_choice_5 = ("⑤" in q_text) or bool(re.search(r"(?:^|\n)\s*5\)", q_text))
                        prev_col_last_q_idx = None if q_has_choice_5 else (len(results) - 1)
                    elif is_last_in_col:
                        prev_col_last_q_idx = None

                # 현재 단의 모든 문항을 먼저 처리한 뒤, 단 하단에 미처 첫 문항을 만나지 못한 [N~M] 지문이 남아 있으면 다음 단으로 이월한다.
                if (
                    pending_passage_y0 is not None
                    and pending_passage_start_q is not None
                    and pending_passage_end_q is not None
                ):
                    p_bottom = 0.895 * page_h
                    p_boxes = [
                        b
                        for b in content_boxes
                        if b[3] > pending_passage_y0 - 4.0 and b[1] < p_bottom
                    ]
                    if p_boxes:
                        p_clip = pymupdf.Rect(
                            max(col_x0 + 2.0, min(b[0] for b in p_boxes) - 10.0),
                            max(0.055 * page_h, min(pending_passage_y0, min(b[1] for b in p_boxes)) - 6.0),
                            min(col_x1 - 2.0, max(b[2] for b in p_boxes) + 10.0),
                            min(0.902 * page_h, max(b[3] for b in p_boxes) + 8.0),
                        )
                        p_pix = page.get_pixmap(dpi=effective_dpi, clip=p_clip, alpha=False)
                        p_txt = "\n".join(
                            ln["text"]
                            for ln in lines
                            if ln["y1"] >= p_clip.y0
                            and ln["y0"] <= p_clip.y1
                            and not _is_footer_or_header_noise(ln["text"], ln["y0"], page_h)
                        )
                        grp_str = f"{pending_passage_start_q}~{pending_passage_end_q}"
                        carried_passage = {
                            "start_q": pending_passage_start_q,
                            "end_q": pending_passage_end_q,
                            "group_id": grp_str,
                            "png_bytes": p_pix.tobytes("png"),
                            "text": p_txt,
                        }
                        active_passage_group = {
                            "start_q": pending_passage_start_q,
                            "end_q": pending_passage_end_q,
                            "group_id": grp_str,
                            "text": p_txt,
                        }
                        prev_col_last_q_idx = None

    results.sort(key=lambda r: r.q_num)
    has_text_layer = total_text_chars >= 60
    return results, has_text_layer
