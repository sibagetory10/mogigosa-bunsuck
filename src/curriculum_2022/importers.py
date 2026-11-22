from __future__ import annotations

import io
import json
import re
import struct
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path
from typing import Any, Iterable

from src.curriculum_2022.catalog import (
    CURRICULUM_CATEGORIES,
    _MOE_SEED_PATH,
    _load_curriculum_2022_catalog_readonly,
    load_curriculum_2022_catalog,
    merge_ncic_standards_to_catalog,
    normalize_subject_2022,
)

_NCIC_STD_LINE_RE = re.compile(r"\[((?:10|12)[가-힣0-9ⅠⅡ]+(?:\d+)?(?:-\d{2}){1,2})\]\s*(.*)")
def _infer_category_from_std(code: str, subject_name: str, fallback_cat: str) -> str:
    txt = f"{code} {subject_name}"
    if any(k in txt for k in ("한사", "한국사")):
        return "한국사"
    if any(k in txt for k in ("공국", "국어", "화언", "화법", "독작", "독서와 작문", "문학", "주독", "문영", "직의", "독토", "매의", "매체 의사소통", "언탐", "언어생활")):
        return "국어"
    if any(k in txt for k in ("공영", "기영", "영어", "영Ⅰ", "영Ⅱ", "영1", "영2", "영독", "영문", "영발", "심영", "심독", "직영", "실영", "미영", "세영")):
        return "영어"
    if any(k in txt for k in ("공수", "수학", "대수", "미적", "확통", "확률", "기하")):
        return "수학"
    if any(k in txt for k in ("통과", "과학", "물리", "화학", "생명", "지구", "역에", "전자", "물질", "화반", "세포", "유전", "지시", "행우")):
        return "과학"
    if any(k in txt for k in ("통사", "사회", "지리", "역사", "윤리", "정치", "법과", "경제", "사문", "세지", "세사", "현윤", "윤사")):
        return "사회"
    return fallback_cat if fallback_cat in CURRICULUM_CATEGORIES else "과학"


def _parse_curriculum_text_lines(
    raw_lines: Iterable[str],
    subject_category_hint: str = "",
    catalog: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """원문 텍스트 줄 목록(PDF/HWP 공통)에서 2022 성취기준 코드·과목·단원·문장을 추출한다."""
    cat_obj = catalog or _load_curriculum_2022_catalog_readonly()
    all_subject_names: list[str] = []
    for cat_entry in cat_obj.get("categories", {}).values():
        for subj in cat_entry.get("subjects") or []:
            if subj.get("name"):
                all_subject_names.append(subj["name"])
    # 긴 과목명 우선 매칭 (예: '통합과학1'이 '통합과학'보다 우선)
    all_subject_names.sort(key=len, reverse=True)

    current_subject = ""
    current_unit = ""
    parsed_items: list[dict[str, Any]] = []
    seen_codes: set[str] = set()
    active_item: dict[str, Any] | None = None

    for raw_line in raw_lines:
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue

        # 과목 제목 줄 감지
        if len(line) <= 35:
            for s_name in all_subject_names:
                if s_name == line or line.endswith(s_name) or re.match(rf"^(?:\d+\.|[가-힣]\.)?\s*\[?{re.escape(s_name)}\]?(?:\s*\(.*\))?$", line):
                    current_subject = s_name
                    break

        # 단원/영역 제목 감지
        if re.match(r"^(?:\(\d+\)|[ⅠⅡⅢⅣⅤⅥⅦⅧ]\.|[1-9]\.)\s*[가-힣A-Za-z0-9·\s]{2,40}$", line):
            cleaned_u = re.sub(r"^(?:\(\d+\)|[ⅠⅡⅢⅣⅤⅥⅦⅧ]\.|[1-9]\.)\s*", "", line).strip()
            if cleaned_u and not any(s in cleaned_u for s in ("성취기준", "교수·학습", "평가")):
                current_unit = cleaned_u

        m = _NCIC_STD_LINE_RE.match(line)
        if m:
            if active_item is not None:
                active_item["statement"] = re.sub(r"\s+", " ", active_item["statement"]).strip()[:320]
                if active_item["code"] not in seen_codes and active_item["statement"]:
                    parsed_items.append(active_item)
                    seen_codes.add(active_item["code"])

            std_code = m.group(1).strip()
            rest = m.group(2).strip()
            grade_int = 1 if std_code.startswith("10") else 2
            inferred_cat = _infer_category_from_std(std_code, current_subject, subject_category_hint)
            norm_subj, _, _, _ = normalize_subject_2022(current_subject, inferred_cat, cat_obj)
            active_item = {
                "code": std_code,
                "category": inferred_cat,
                "subject": norm_subj or current_subject,
                "unit": current_unit,
                "min_grade": grade_int,
                "statement": rest,
            }
            continue

        if active_item is not None:
            # 해설/다음 섹션 헤더가 나오면 현재 성취기준 문장 수집 종료
            if line.startswith(("<성취기준 해설>", "(가)", "(나)", "[성취기준 해설]", "교수·학습", "평가")):
                active_item["statement"] = re.sub(r"\s+", " ", active_item["statement"]).strip()[:320]
                if active_item["code"] not in seen_codes and active_item["statement"]:
                    parsed_items.append(active_item)
                    seen_codes.add(active_item["code"])
                active_item = None
            elif len(active_item["statement"]) < 260 and not line.startswith(("•", "◦", "※")):
                active_item["statement"] += " " + line

    if active_item is not None:
        active_item["statement"] = re.sub(r"\s+", " ", active_item["statement"]).strip()[:320]
        if active_item["code"] not in seen_codes and active_item["statement"]:
            parsed_items.append(active_item)

    return parsed_items


def parse_ncic_curriculum_pdf(
    pdf_bytes: bytes,
    subject_category_hint: str = "",
    catalog: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    NCIC(국가교육과정정보센터) 교육부 고시 제2022-33호 별책 PDF에서
    성취기준 코드([10...]/[12...]), 과목명, 권장 학년, 성취기준 문장을 추출한다.
    """
    import pymupdf

    lines: list[str] = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            raw_text = page.get_text("text") or ""
            lines.extend(raw_text.splitlines())
    return _parse_curriculum_text_lines(lines, subject_category_hint=subject_category_hint, catalog=catalog)
_MOE_2022_33_ZIP_URL = "https://www.moe.go.kr/boardCnts/fileDown.do?m=&s=moe&fileSeq=1512facbda6c234a1641ac7e9c156ca2"
_MOE_2022_33_REFERER = "https://www.moe.go.kr/boardCnts/viewRenew.do?boardID=141&boardSeq=93458&lev=0&searchType=null&statusYN=W"


def _read_ole2_streams(data: bytes) -> dict[str, bytes]:
    """외부 의존성 없이 HWP5(OLE2/CFBF) 바이너리 스트림을 읽는다."""
    if data[:8] != b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        raise ValueError("HWP5(OLE2) 형식이 아닙니다.")
    sector_size = 1 << struct.unpack_from("<H", data, 0x1E)[0]
    mini_sector_size = 1 << struct.unpack_from("<H", data, 0x20)[0]
    fat_count = struct.unpack_from("<I", data, 0x2C)[0]
    first_dir_sector = struct.unpack_from("<I", data, 0x30)[0]
    mini_cutoff = struct.unpack_from("<I", data, 0x38)[0]
    first_minifat_sector = struct.unpack_from("<I", data, 0x3C)[0]
    minifat_count = struct.unpack_from("<I", data, 0x40)[0]
    first_difat_sector = struct.unpack_from("<I", data, 0x44)[0]
    difat_count = struct.unpack_from("<I", data, 0x48)[0]

    def read_sector(sec_id: int) -> bytes:
        off = (sec_id + 1) * sector_size
        return data[off : off + sector_size]

    fat_sectors = [s for s in struct.unpack_from("<109I", data, 0x4C) if s < 0xFFFFFFFC]
    difat_sec = first_difat_sector
    for _ in range(difat_count):
        if difat_sec >= 0xFFFFFFFC:
            break
        buf = read_sector(difat_sec)
        entries = struct.unpack_from(f"<{sector_size // 4}I", buf, 0)
        fat_sectors.extend(s for s in entries[:-1] if s < 0xFFFFFFFC)
        difat_sec = entries[-1]
    fat_sectors = fat_sectors[:fat_count]

    fat: list[int] = []
    for fs in fat_sectors:
        fat.extend(struct.unpack_from(f"<{sector_size // 4}I", read_sector(fs), 0))

    def read_chain(start: int, size: int | None = None) -> bytes:
        out = bytearray()
        sec = start
        seen: set[int] = set()
        while sec < 0xFFFFFFFC and sec not in seen:
            seen.add(sec)
            out.extend(read_sector(sec))
            sec = fat[sec] if sec < len(fat) else 0xFFFFFFFE
        return bytes(out[:size] if size is not None else out)

    dir_bytes = read_chain(first_dir_sector)
    entries = []
    for i in range(0, len(dir_bytes), 128):
        ent = dir_bytes[i : i + 128]
        if len(ent) < 128:
            break
        name_len = struct.unpack_from("<H", ent, 0x40)[0]
        obj_type = ent[0x42]
        name = ent[: max(0, name_len - 2)].decode("utf-16le", "ignore") if name_len >= 2 else ""
        start_sec = struct.unpack_from("<I", ent, 0x74)[0]
        stream_size = struct.unpack_from("<Q", ent, 0x78)[0] & 0xFFFFFFFF
        entries.append((name, obj_type, start_sec, stream_size))

    mini_stream = b""
    minifat: list[int] = []
    if entries and entries[0][1] == 5 and entries[0][2] < 0xFFFFFFFC:
        mini_stream = read_chain(entries[0][2], entries[0][3])
    if first_minifat_sector < 0xFFFFFFFC and minifat_count > 0:
        mf_bytes = read_chain(first_minifat_sector)
        minifat = list(struct.unpack_from(f"<{len(mf_bytes) // 4}I", mf_bytes, 0))

    def read_mini_chain(start: int, size: int) -> bytes:
        out = bytearray()
        sec = start
        seen: set[int] = set()
        while sec < 0xFFFFFFFC and sec not in seen and len(out) < size:
            seen.add(sec)
            off = sec * mini_sector_size
            out.extend(mini_stream[off : off + mini_sector_size])
            sec = minifat[sec] if sec < len(minifat) else 0xFFFFFFFE
        return bytes(out[:size])

    streams: dict[str, bytes] = {}
    for name, obj_type, start_sec, stream_size in entries:
        if obj_type == 2 and stream_size > 0 and start_sec < 0xFFFFFFFC:
            streams[name] = (
                read_mini_chain(start_sec, stream_size)
                if (stream_size < mini_cutoff and mini_stream)
                else read_chain(start_sec, stream_size)
            )
    return streams


def _extract_hwp_text(hwp_bytes: bytes) -> str:
    """HWP5 BodyText/Section* 스트림에서 문단 텍스트를 추출한다."""
    streams = _read_ole2_streams(hwp_bytes)
    hdr = streams.get("FileHeader", b"")
    compressed = bool(len(hdr) > 36 and (hdr[36] & 1))
    lines: list[str] = []
    for sname in sorted(k for k in streams if k.startswith("Section")):
        raw = streams[sname]
        if compressed:
            try:
                raw = zlib.decompress(raw, -15)
            except Exception:
                continue
        pos = 0
        while pos + 4 <= len(raw):
            header = struct.unpack_from("<I", raw, pos)[0]
            pos += 4
            tag_id = header & 0x3FF
            size = (header >> 20) & 0xFFF
            if size == 0xFFF and pos + 4 <= len(raw):
                size = struct.unpack_from("<I", raw, pos)[0]
                pos += 4
            rec = raw[pos : pos + size]
            pos += size
            if tag_id == 67:  # HWPTAG_PARA_TEXT
                chars: list[str] = []
                i = 0
                while i + 2 <= len(rec):
                    code = struct.unpack_from("<H", rec, i)[0]
                    if code in (1, 2, 3, 11, 12, 14, 15, 16, 17, 18, 21, 22, 23):
                        i += 16
                        continue
                    if code in (4, 5, 6, 7, 8, 9, 19, 20):
                        i += 2
                        continue
                    if code in (10, 13):
                        chars.append("\n")
                    elif code >= 32:
                        chars.append(chr(code))
                    i += 2
                txt = "".join(chars).strip()
                if txt:
                    lines.append(txt)
    return "\n".join(lines)


def _download_moe_zip(timeout: int = 8, total_timeout: int = 15) -> bytes:
    """교육부 원문 ZIP을 전체 시간 상한(total_timeout) 내에 내려받는다."""
    req = urllib.request.Request(
        _MOE_2022_33_ZIP_URL,
        headers={"User-Agent": "Mozilla/5.0", "Referer": _MOE_2022_33_REFERER},
    )
    deadline = time.monotonic() + max(2, int(total_timeout))
    chunks: list[bytes] = []
    with urllib.request.urlopen(req, timeout=max(2, int(timeout))) as resp:
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(f"교육부(moe.go.kr) 다운로드 제한시간({total_timeout}초) 초과")
            chunk = resp.read(65536)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks)


def _extract_moe_zip_items(
    zip_bytes: bytes,
    target_set: set[str],
    catalog: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """교육부 고시 제2022-33호 ZIP 바이너리에서 별책 HWP 성취기준을 추출한다."""
    hint_rules: list[tuple[str, str]] = [
        ("국어과", "국어"),
        ("영어과", "영어"),
        ("수학과", "수학"),
        ("사회과", "사회"),
        ("과학과", "과학"),
    ]
    all_extracted: list[dict[str, Any]] = []
    per_file_counts: dict[str, int] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            try:
                fname = info.filename.encode("cp437").decode("cp949")
            except Exception:
                fname = info.filename
            matched_hint = next((cat for tok, cat in hint_rules if tok in fname), "")
            if not matched_hint:
                continue
            # 사회과 별책 7에는 한국사도 포함되므로 사회 또는 한국사 요청 시 모두 파싱
            if matched_hint not in target_set and not (matched_hint == "사회" and "한국사" in target_set):
                continue
            raw_hwp = zf.read(info.filename)
            txt = _extract_hwp_text(raw_hwp)
            items = _parse_curriculum_text_lines(
                txt.splitlines(),
                subject_category_hint=matched_hint,
                catalog=catalog,
            )
            filtered_items = [it for it in items if it.get("category") in target_set]
            per_file_counts[fname] = len(filtered_items)
            all_extracted.extend(filtered_items)
    return all_extracted, per_file_counts


def fetch_and_sync_moe_2022_curriculum(
    target_categories: Sequence[str] | None = None,
    path: Path | None = None,
    timeout: int = 8,
    total_timeout: int = 15,
    allow_seed_fallback: bool = True,
) -> dict[str, Any]:
    """
    교육부(moe.go.kr) 고시 제2022-33호 공식 원문 ZIP(별책 5·7·8·9·14)을 웹에서 직접 내려받아
    국어·영어·수학·사회·과학·한국사 성취기준을 자동 파싱·병합한다.
    해외 서버 IP 차단/지연 등으로 웹 접속이 제한시간 내에 실패하면 내장 공식 원문 시드로 즉시 폴백한다.
    """
    cat_obj = load_curriculum_2022_catalog()
    target_set = set(target_categories) if target_categories else set(CURRICULUM_CATEGORIES)

    source_mode = "web"
    fallback_reason = ""
    all_extracted: list[dict[str, Any]] = []
    per_file_counts: dict[str, int] = {}

    try:
        zip_bytes = _download_moe_zip(timeout=timeout, total_timeout=total_timeout)
        all_extracted, per_file_counts = _extract_moe_zip_items(zip_bytes, target_set, cat_obj)
        if not all_extracted:
            raise ValueError("교육부 원문 ZIP에서 추출된 성취기준이 0건입니다.")
    except Exception as exc:
        if not (allow_seed_fallback and _MOE_SEED_PATH.exists()):
            raise
        seed_raw = json.loads(_MOE_SEED_PATH.read_text(encoding="utf-8"))
        seed_items = seed_raw.get("items") or []
        all_extracted = [it for it in seed_items if it.get("category") in target_set]
        per_file_counts = dict(seed_raw.get("files") or {})
        source_mode = "seed_fallback"
        fallback_reason = str(exc)

    res = merge_ncic_standards_to_catalog(all_extracted, path=path)
    return {
        "extracted_total": len(all_extracted),
        "added": res["added"],
        "updated": res["updated"],
        "files": per_file_counts,
        "source": source_mode,
        "fallback_reason": fallback_reason,
    }
