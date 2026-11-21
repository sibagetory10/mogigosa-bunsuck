from __future__ import annotations

import copy
import io
import json
import re
import struct
import urllib.request
import zipfile
import zlib
from pathlib import Path
from typing import Any, Iterable, Sequence

import pymupdf

from src.models import ConceptBookRecord, QuestionExtractionOutput

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "curriculum_2022.json"
_RUNTIME_STANDARDS_PATH = Path(__file__).resolve().parent.parent / "exam_db" / "ncic_standards.json"
_CATALOG_CACHE: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}


def _stat_sig(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
        return (int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        return None

_EVERYDAY_TERM_STOPWORDS = frozenset(
    {
        "그림",
        "표",
        "자료",
        "보기",
        "다음",
        "문항",
        "문제",
        "정답",
        "해설",
        "과정",
        "결과",
        "값",
        "합",
        "차",
        "곱",
        "비",
        "수",
        "점",
        "선",
        "면",
        "ㄱ",
        "ㄴ",
        "ㄷ",
        "ㄹ",
        "개",
        "번",
        "명",
        "회",
        "가지",
        "원",
        "cm",
        "mm",
        "km",
        "g",
        "kg",
        "mg",
        "ml",
        "mL",
        "L",
        "s",
        "sec",
        "min",
        "h",
    }
)

_NCIC_STD_LINE_RE = re.compile(r"\[((?:10|12)[가-힣0-9ⅠⅡ]+(?:\d+)?(?:-\d{2}){1,2})\]\s*(.*)")
_STD_CODE_ONLY_RE = re.compile(r"^\[?(?:10|12)[가-힣0-9ⅠⅡ]+(?:\d+)?(?:-\d{2}){1,2}\]?$")


def _canon_subject_key(text: str) -> str:
    s = (text or "").strip().lower()
    s = s.replace("ⅱ", "2").replace("ⅰ", "1").replace("Ⅱ", "2").replace("Ⅰ", "1")
    s = re.sub(r"([가-힣])ii(?=$|\s)", r"\g<1>2", s)
    s = re.sub(r"([가-힣])i(?=$|\s)", r"\g<1>1", s)
    s = re.sub(r"[\s·ㆍ・\-_().]+", "", s)
    return s


def _load_runtime_standards(runtime_path: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """Git 비추적 런타임 저장소(exam_db/ncic_standards.json)에서 업로드된 성취기준을 로드한다."""
    target = runtime_path or _RUNTIME_STANDARDS_PATH
    if not target.exists():
        return {}
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
        cats = raw.get("categories") if isinstance(raw, dict) and "categories" in raw else raw
        if not isinstance(cats, dict):
            return {}
        return {
            str(k): list(v)
            for k, v in cats.items()
            if isinstance(v, list)
        }
    except Exception:
        return {}


def _save_runtime_standards(
    standards_by_cat: dict[str, list[dict[str, Any]]],
    runtime_path: Path | None = None,
) -> None:
    """업로드된 NCIC 성취기준을 Git 비추적 런타임 저장소(exam_db/ncic_standards.json)에 저장한다."""
    target = runtime_path or _RUNTIME_STANDARDS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"version": "2022-33", "categories": standards_by_cat}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _CATALOG_CACHE.clear()


def load_curriculum_2022_catalog(path: Path | None = None) -> dict[str, Any]:
    """2022 개정 교육과정 공식 편제 및 NCIC 누적 성취기준 JSON을 로드한다 (mtime 기반 메모리 캐시 적용)."""
    target = path or _DATA_PATH
    if path is None:
        cache_key = f"default::{target}::{_RUNTIME_STANDARDS_PATH}"
        sig = (_stat_sig(target), _stat_sig(_RUNTIME_STANDARDS_PATH))
    else:
        cache_key = f"custom::{target}"
        sig = (_stat_sig(target),)

    cached = _CATALOG_CACHE.get(cache_key)
    if cached is not None and cached[0] == sig:
        return copy.deepcopy(cached[1])

    if not target.exists():
        catalog: dict[str, Any] = {"version": "2022-33", "categories": {}}
    else:
        try:
            catalog = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            catalog = {"version": "2022-33", "categories": {}}

    if path is None:
        runtime_stds = _load_runtime_standards()
        cat_map = catalog.setdefault("categories", {})
        for cat_name, std_list in runtime_stds.items():
            if not std_list:
                continue
            cat_entry = cat_map.setdefault(
                cat_name,
                {"subjects": [], "shift_notes": [], "core_units": {}, "standards": []},
            )
            merged_by_code: dict[str, dict[str, Any]] = {
                (s.get("code") or "").strip(): s
                for s in (cat_entry.get("standards") or [])
                if (s.get("code") or "").strip()
            }
            for item in std_list:
                code = (item.get("code") or "").strip()
                if code:
                    merged_by_code[code] = item
            cat_entry["standards"] = list(merged_by_code.values())

    _CATALOG_CACHE[cache_key] = (sig, copy.deepcopy(catalog))
    return catalog


def save_curriculum_2022_catalog(catalog: dict[str, Any], path: Path | None = None) -> None:
    """갱신된 2022 개정 교육과정 기준표를 JSON 파일로 저장한다."""
    target = path or _DATA_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _CATALOG_CACHE.clear()


CURRICULUM_CATEGORIES: tuple[str, ...] = ("국어", "영어", "수학", "사회", "과학", "한국사")


def get_catalog_summary_stats(catalog: dict[str, Any] | None = None) -> dict[str, dict[str, int]]:
    """대분류(국어·영어·수학·사회·과학·한국사)별 공식 과목 수 및 NCIC 성취기준 수 통계를 반환한다."""
    cat_data = (catalog or load_curriculum_2022_catalog()).get("categories", {})
    stats: dict[str, dict[str, int]] = {}
    for cat_name in CURRICULUM_CATEGORIES:
        entry = cat_data.get(cat_name, {})
        stats[cat_name] = {
            "subjects": len(entry.get("subjects") or []),
            "standards": len(entry.get("standards") or []),
        }
    return stats


def get_official_subjects_for_category(
    subject_category: str,
    catalog: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cat_data = (catalog or load_curriculum_2022_catalog()).get("categories", {})
    entry = cat_data.get(subject_category, {})
    return list(entry.get("subjects") or [])


def normalize_subject_2022(
    raw_subject: str,
    subject_category: str = "",
    catalog: dict[str, Any] | None = None,
) -> tuple[str, bool, dict[str, Any] | None, bool]:
    """
    추출된 세부 과목명을 2022 개정 공식 과목명으로 정규화한다.
    반환: (정규화된_과목명, 공식_편제_일치_여부, 과목_메타_딕셔너리, 별칭_자동교정_여부)
    """
    cat_map = (catalog or load_curriculum_2022_catalog()).get("categories", {})
    cleaned = re.sub(r"\s+", " ", (raw_subject or "").strip())
    if not cleaned:
        return "", False, None, False

    raw_key = _canon_subject_key(cleaned)

    search_cats: list[str] = []
    if subject_category in cat_map:
        search_cats.append(subject_category)
    for c in CURRICULUM_CATEGORIES:
        if c not in search_cats and c in cat_map:
            search_cats.append(c)

    for cat in search_cats:
        for subj in cat_map[cat].get("subjects") or []:
            official_name = subj.get("name", "")
            if _canon_subject_key(official_name) == raw_key:
                was_changed = official_name != cleaned
                return official_name, True, subj, was_changed

    for cat in search_cats:
        for subj in cat_map[cat].get("subjects") or []:
            official_name = subj.get("name", "")
            for alias in subj.get("aliases") or []:
                if _canon_subject_key(alias) == raw_key:
                    return official_name, True, subj, True

    return cleaned, False, None, False


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
    cat_obj = catalog or load_curriculum_2022_catalog()
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


def fetch_and_sync_moe_2022_curriculum(
    target_categories: Sequence[str] | None = None,
    path: Path | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    """
    교육부(moe.go.kr) 고시 제2022-33호 공식 원문 ZIP(별책 5·7·8·9·14)을 웹에서 직접 내려받아
    국어·영어·수학·사회·과학·한국사 성취기준을 자동 파싱·병합한다.
    """
    req = urllib.request.Request(
        _MOE_2022_33_ZIP_URL,
        headers={"User-Agent": "Mozilla/5.0", "Referer": _MOE_2022_33_REFERER},
    )
    zip_bytes = urllib.request.urlopen(req, timeout=timeout).read()
    cat_obj = load_curriculum_2022_catalog()
    target_set = set(target_categories) if target_categories else set(CURRICULUM_CATEGORIES)

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
                catalog=cat_obj,
            )
            filtered_items = [it for it in items if it.get("category") in target_set]
            per_file_counts[fname] = len(filtered_items)
            all_extracted.extend(filtered_items)

    res = merge_ncic_standards_to_catalog(all_extracted, path=path)
    return {
        "extracted_total": len(all_extracted),
        "added": res["added"],
        "updated": res["updated"],
        "files": per_file_counts,
    }


def merge_ncic_standards_to_catalog(
    extracted_standards: list[dict[str, Any]],
    path: Path | None = None,
) -> dict[str, int]:
    """추출된 NCIC 성취기준 리스트를 병합 저장한다. (기본 경로는 Git 비추적 exam_db/ncic_standards.json)"""
    added = 0
    updated = 0

    if path is None:
        runtime_stds = _load_runtime_standards()
        for item in extracted_standards:
            code = (item.get("code") or "").strip()
            cat = (item.get("category") or "").strip()
            if not code or cat not in CURRICULUM_CATEGORIES:
                continue
            stds = runtime_stds.setdefault(cat, [])
            existing_idx = next((i for i, s in enumerate(stds) if s.get("code") == code), None)
            clean_entry = {
                "code": code,
                "subject": (item.get("subject") or "").strip(),
                "unit": (item.get("unit") or "").strip(),
                "min_grade": int(item.get("min_grade") or (1 if code.startswith("10") else 2)),
                "statement": (item.get("statement") or "").strip(),
            }
            if existing_idx is None:
                stds.append(clean_entry)
                added += 1
            else:
                stds[existing_idx] = clean_entry
                updated += 1
        _save_runtime_standards(runtime_stds)
        return {"added": added, "updated": updated}

    catalog = load_curriculum_2022_catalog(path)
    cat_map = catalog.setdefault("categories", {})

    for item in extracted_standards:
        code = (item.get("code") or "").strip()
        cat = (item.get("category") or "").strip()
        if not code or cat not in CURRICULUM_CATEGORIES:
            continue
        cat_entry = cat_map.setdefault(cat, {"subjects": [], "shift_notes": [], "core_units": {}, "standards": []})
        stds: list[dict[str, Any]] = cat_entry.setdefault("standards", [])
        existing_idx = next((i for i, s in enumerate(stds) if s.get("code") == code), None)
        clean_entry = {
            "code": code,
            "subject": (item.get("subject") or "").strip(),
            "unit": (item.get("unit") or "").strip(),
            "min_grade": int(item.get("min_grade") or (1 if code.startswith("10") else 2)),
            "statement": (item.get("statement") or "").strip(),
        }
        if existing_idx is None:
            stds.append(clean_entry)
            added += 1
        else:
            stds[existing_idx] = clean_entry
            updated += 1

    save_curriculum_2022_catalog(catalog, path)
    return {"added": added, "updated": updated}


def clear_ncic_standards_for_category(
    category: str,
    path: Path | None = None,
) -> int:
    """특정 교과 대분류에 누적 저장된 NCIC 성취기준(standards)만 초기화하고 삭제된 개수를 반환한다."""
    if category not in CURRICULUM_CATEGORIES:
        return 0
    if path is None:
        runtime_stds = _load_runtime_standards()
        existing = runtime_stds.get(category) or []
        removed_count = len(existing)
        if removed_count > 0:
            runtime_stds[category] = []
            _save_runtime_standards(runtime_stds)
        return removed_count
    catalog = load_curriculum_2022_catalog(path)
    cat_map = catalog.setdefault("categories", {})
    cat_entry = cat_map.get(category)
    if not cat_entry:
        return 0
    existing = cat_entry.get("standards") or []
    removed_count = len(existing)
    if removed_count > 0:
        cat_entry["standards"] = []
        save_curriculum_2022_catalog(catalog, path)
    return removed_count


def has_ncic_standards_for_category(
    subject_category: str,
    catalog: dict[str, Any] | None = None,
) -> bool:
    """해당 교과 대분류에 NCIC 고시 성취기준이 1개 이상 등록되어 있는지 반환한다."""
    cat_obj = catalog or load_curriculum_2022_catalog()
    entry = cat_obj.get("categories", {}).get(subject_category, {})
    return bool(entry.get("standards"))


def validate_standards_only(
    standard_code: str,
    standard_codes: Sequence[str],
    subject_category: str,
    catalog: dict[str, Any] | None = None,
) -> tuple[str, list[str], bool, list[str]]:
    """
    국어·영어 등 별도 스키마 교과용 성취기준 코드 정제 및 할루시네이션(가짜 코드) 검증 헬퍼.
    반환: (정제된_대표코드, 정제된_코드목록, 검토필요_여부, 검토_노트)
    """
    cat_obj = catalog or load_curriculum_2022_catalog()
    cat_entry = cat_obj.get("categories", {}).get(subject_category, {})
    loaded_stds = {
        (s.get("code") or "").strip(): s
        for s in (cat_entry.get("standards") or [])
        if s.get("code")
    }

    std_code = re.sub(r"[\[\]\s]+", "", standard_code or "")
    cleaned_codes: list[str] = []
    seen_codes: set[str] = set()
    for rc in ([std_code] if std_code else []) + list(standard_codes or []):
        c = re.sub(r"[\[\]\s]+", "", rc or "")
        if c and c not in seen_codes:
            seen_codes.add(c)
            cleaned_codes.append(c)

    if not std_code and cleaned_codes:
        std_code = cleaned_codes[0]

    needs_review = False
    review_notes: list[str] = []
    if loaded_stds and cleaned_codes:
        fake_codes = [c for c in cleaned_codes if _STD_CODE_ONLY_RE.match(c) and c not in loaded_stds]
        if fake_codes:
            needs_review = True
            joined_fake = ", ".join(f"[{c}]" for c in fake_codes)
            review_notes.append(f"로드된 NCIC 공식 성취기준 목록에 없는 코드입니다(할루시네이션 의심): {joined_fake}")

    return std_code, cleaned_codes, needs_review, review_notes


def _infer_target_subject_from_exam_name(exam_name: str, cat_entry: dict[str, Any]) -> str:
    """시험 회차명(예: '2026년 3월 고3 물리학1')에서 2022 공식 과목명을 정밀 추론한다."""
    canon_exam = _canon_subject_key(exam_name)
    if not canon_exam:
        return ""
    candidates: list[tuple[str, str]] = []
    for subj in cat_entry.get("subjects") or []:
        official_name = (subj.get("name") or "").strip()
        if not official_name:
            continue
        candidates.append((_canon_subject_key(official_name), official_name))
        for alias in subj.get("aliases") or []:
            ck = _canon_subject_key(alias)
            if ck:
                candidates.append((ck, official_name))
    # 긴 과목/별칭 우선 매칭 ('미적분2'가 '미적분'보다 우선)
    candidates.sort(key=lambda x: len(x[0]), reverse=True)
    for key, official_name in candidates:
        if key and key in canon_exam:
            return official_name
    return ""


def build_tagging_grounding_context(
    subject_category: str,
    exam_name: str = "",
    concept_books: Sequence[ConceptBookRecord] | None = None,
    max_standards: int = 45,
    max_book_units: int = 28,
    catalog: dict[str, Any] | None = None,
) -> str:
    """
    Tab 1 태깅 프롬프트에 주입할 2022 개정 공식 기준표 + NCIC 성취기준 + 공용 개념서 DB 요약 컨텍스트를 조립한다.
    현재 시험지의 과목이 판별되면 해당 과목과 고1 공통과목만 동적 필터링해 프롬프트 토큰을 절감한다.
    """
    cat_obj = catalog or load_curriculum_2022_catalog()
    cat_data = cat_obj.get("categories", {})
    entry = cat_data.get(subject_category)
    if not entry:
        return ""

    target_subject = _infer_target_subject_from_exam_name(exam_name, entry)
    common_subjects = {
        (s.get("name") or "").strip()
        for s in (entry.get("subjects") or [])
        if s.get("track") == "공통" and s.get("name")
    }
    allowed_scope_subjects: set[str] | None = (
        ({target_subject} | common_subjects) if target_subject else None
    )

    header = f"[2022 개정 교육과정 공식 기준표 — 대분류: {subject_category}"
    if target_subject:
        header += f" · 시험 과목 필터: {target_subject}"
    header += "]"
    blocks: list[str] = [header]

    # 1) 과목 편제표 (구분별 공식 과목명)
    by_track: dict[str, list[str]] = {}
    for subj in entry.get("subjects") or []:
        track = subj.get("track", "선택")
        by_track.setdefault(track, []).append(subj.get("name", ""))
    for track, names in by_track.items():
        if names:
            grade_hint = "고1 권장" if track == "공통" else "고2~3 선택"
            blocks.append(f"- {track}({grade_hint}): {', '.join(names)}")

    # 2) 2015→2022 개정 주요 변경 주의사항
    shifts = entry.get("shift_notes") or []
    if shifts:
        blocks.append("\n[2015→2022 개정 이동 주의사항 (할루시네이션 방지)]")
        for note in shifts[:5]:
            blocks.append(f"- {note}")

    # 3) 과목별 핵심 단원 개요 (대상 과목이 판별되면 해당 과목 + 고1 공통만 포함)
    core_units = entry.get("core_units") or {}
    if core_units:
        filtered_units = {
            s_name: u_list
            for s_name, u_list in core_units.items()
            if allowed_scope_subjects is None or s_name in allowed_scope_subjects
        }
        if filtered_units:
            blocks.append("\n[2022 관련 과목 단원 체계]")
            for s_name, u_list in filtered_units.items():
                blocks.append(f"- {s_name}: {', '.join(u_list)}")

    # 4) NCIC 공식 고시 PDF에서 파싱된 성취기준 (대상 과목 + 고1 공통만 동적 필터링)
    standards = entry.get("standards") or []
    if standards:
        primary_stds: list[dict[str, Any]] = []
        common_stds: list[dict[str, Any]] = []
        fallback_stds: list[dict[str, Any]] = []
        for st_item in standards:
            raw_st_subj = st_item.get("subject", "")
            norm_st_subj, _, _, _ = normalize_subject_2022(raw_st_subj, subject_category, cat_obj)
            eff_st_subj = norm_st_subj or raw_st_subj
            if target_subject and eff_st_subj == target_subject:
                primary_stds.append(st_item)
            elif eff_st_subj in common_subjects or st_item.get("min_grade") == 1:
                common_stds.append(st_item)
            elif allowed_scope_subjects is None:
                fallback_stds.append(st_item)

        chosen = (primary_stds + common_stds + fallback_stds)[:max_standards]
        if chosen:
            blocks.append(f"\n[NCIC 공식 고시 성취기준 발췌 ({len(chosen)}/{len(standards)}개 · 과목 필터링 적용)]")
            for st_item in chosen:
                blocks.append(
                    f"- [{st_item.get('code')}] ({st_item.get('subject')} · 고{st_item.get('min_grade')}) "
                    f"{st_item.get('statement', '')[:120]}"
                )

    # 5) 공용 개념서 DB(concept_db) 단원·허용 용어 연동 (대상 과목 + 고1 공통 교재만 선택)
    if concept_books:
        matching_books: list[ConceptBookRecord] = []
        cat_subject_tokens = {subj.get("name", "") for subj in (entry.get("subjects") or []) if subj.get("name")}
        for bk in concept_books:
            bk_blob = f"{bk.book_title} {bk.subject}"
            in_category = subject_category in bk_blob or any(tok in bk_blob for tok in cat_subject_tokens)
            if not in_category:
                continue
            if allowed_scope_subjects is not None:
                bk_subj_norm, bk_ok, _, _ = normalize_subject_2022(bk.subject, subject_category, cat_obj)
                if not bk_ok:
                    bk_subj_norm = _infer_target_subject_from_exam_name(bk_blob, entry)
                is_grade1_book = "1" in (bk.grade or "")
                if bk_subj_norm and bk_subj_norm not in allowed_scope_subjects:
                    continue
                if not bk_subj_norm and not is_grade1_book and target_subject not in bk_blob:
                    continue
            matching_books.append(bk)

        if matching_books:
            blocks.append("\n[등록된 공용 개념서 DB(concept_db) 단원 및 허용 용어]")
            unit_budget = max_book_units
            for bk in matching_books[:4]:
                if unit_budget <= 0:
                    break
                blocks.append(f"• {bk.book_title} ({bk.grade} · {bk.subject}):")
                for u in bk.units[: min(len(bk.units), unit_budget)]:
                    concepts_str = ", ".join(u.core_concepts[:5])
                    terms_str = ", ".join(u.allowed_terms_and_units[:8])
                    blocks.append(f"  - {u.unit_name}: 원리[{concepts_str}] / 허용용어[{terms_str}]")
                    unit_budget -= 1

    return "\n".join(blocks)


def _clean_term_list(terms: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in terms:
        t = re.sub(r"\s+", " ", (raw or "").strip())
        if not t or t in _EVERYDAY_TERM_STOPWORDS or t.lower() in _EVERYDAY_TERM_STOPWORDS:
            continue
        if len(t) == 1 and t.isdigit():
            continue
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _collect_lower_grade_allowed_vocab(
    subject_category: str,
    min_grade: int,
    concept_books: Sequence[ConceptBookRecord] | None,
) -> set[str]:
    vocab: set[str] = set()
    if not concept_books:
        return vocab
    for bk in concept_books:
        m = re.search(r"[123]", bk.grade or "")
        bk_grade = int(m.group(0)) if m else 3
        if bk_grade <= min_grade:
            for u in bk.units:
                for term in u.allowed_terms_and_units:
                    clean_t = (term or "").strip()
                    if clean_t:
                        vocab.add(clean_t.lower())
    return vocab


def validate_and_normalize_extracted_tag(
    extracted: QuestionExtractionOutput,
    subject_category: str,
    concept_books: Sequence[ConceptBookRecord] | None = None,
    catalog: dict[str, Any] | None = None,
) -> QuestionExtractionOutput:
    """
    Gemini 태깅 출력을 2022 개정 공식 기준표 및 공용 개념서 DB와 대조해
    1) 2015→2022 과목명 자동 정규화
    2) 기본 용어(basic_terms) vs 상위 전문 용어(advanced_terms) 분리 및 일상어 필터링
    3) 과목-학년 모순 및 가짜 성취기준 코드 할루시네이션 검출(needs_review)
    을 수행한다.
    """
    cat_obj = catalog or load_curriculum_2022_catalog()
    cat_entry = cat_obj.get("categories", {}).get(subject_category, {})
    loaded_stds = {
        (s.get("code") or "").strip(): s
        for s in (cat_entry.get("standards") or [])
        if s.get("code")
    }

    review_notes: list[str] = list(getattr(extracted, "review_notes", []) or [])
    needs_review = bool(getattr(extracted, "needs_review", False))

    norm_subj, is_official, subj_meta, was_alias = normalize_subject_2022(
        extracted.curriculum_subject,
        subject_category=subject_category,
        catalog=cat_obj,
    )
    if norm_subj:
        extracted.curriculum_subject = norm_subj
    if was_alias:
        review_notes.append(f"2022 공식 과목명으로 자동 정규화됨: {norm_subj}")
    if not is_official:
        needs_review = True
        review_notes.append(f"2022 '{subject_category}' 공식 과목 목록에서 확인되지 않은 과목명입니다: {extracted.curriculum_subject}")

    # 복수 성취기준 코드 정제·동기화 및 할루시네이션 검출
    std_code = re.sub(r"[\[\]\s]+", "", getattr(extracted, "standard_code", "") or "")
    raw_codes = list(getattr(extracted, "standard_codes", []) or [])
    cleaned_codes: list[str] = []
    seen_codes: set[str] = set()
    for rc in ([std_code] if std_code else []) + raw_codes:
        c = re.sub(r"[\[\]\s]+", "", rc or "")
        if c and c not in seen_codes:
            seen_codes.add(c)
            cleaned_codes.append(c)

    if not std_code and cleaned_codes:
        std_code = cleaned_codes[0]

    extracted.standard_code = std_code
    extracted.standard_codes = cleaned_codes

    if loaded_stds:
        fake_codes = [c for c in cleaned_codes if _STD_CODE_ONLY_RE.match(c) and c not in loaded_stds]
        if fake_codes:
            needs_review = True
            joined_fake = ", ".join(f"[{c}]" for c in fake_codes)
            review_notes.append(f"로드된 NCIC 공식 성취기준 목록에 없는 코드입니다(할루시네이션 의심): {joined_fake}")

    # 과목 편제(공통 vs 선택) ↔ 최저 학년 일관성 점검
    grade_evidence = (getattr(extracted, "grade_evidence", "") or "").strip()
    extracted.grade_evidence = grade_evidence
    if subj_meta:
        track = subj_meta.get("track", "")
        min_allowed = int(subj_meta.get("min_allowed_grade") or 1)
        if track == "공통" and extracted.min_grade > 1:
            needs_review = True
            review_notes.append(f"'{extracted.curriculum_subject}'는 고1 공통과목이나 min_grade={extracted.min_grade}로 판정되어 확인이 필요합니다.")
        elif min_allowed >= 2 and extracted.min_grade == 1:
            has_only_grade2_codes = bool(cleaned_codes) and all(c.startswith("12") for c in cleaned_codes)
            if has_only_grade2_codes:
                needs_review = True
                joined_c = ", ".join(f"[{c}]" for c in cleaned_codes)
                review_notes.append(
                    f"성취기준 코드({joined_c})는 고2~3 선택과목 범위인데 min_grade=1로 판정되어 모순이 있습니다."
                )
            elif not grade_evidence:
                needs_review = True
                review_notes.append(
                    f"선택과목('{extracted.curriculum_subject}') 문항을 고1(min_grade=1)로 판정했으나 고1 공통 원리 근거가 비어 있습니다."
                )

    if not extracted.core_principles:
        needs_review = True
        review_notes.append("핵심 풀이 원리(core_principles)가 비어 있습니다.")

    # 용어 정제: everyday 불용어 제거 + concept_db 저학년 허용 용어는 basic_terms로 교정
    adv_clean = _clean_term_list(getattr(extracted, "advanced_terms", []) or [])
    basic_clean = _clean_term_list(getattr(extracted, "basic_terms", []) or [])
    req_clean = _clean_term_list(extracted.required_terms or [])

    lower_vocab = _collect_lower_grade_allowed_vocab(subject_category, extracted.min_grade, concept_books)
    if lower_vocab and adv_clean:
        rebalanced_adv: list[str] = []
        for term in adv_clean:
            if term.lower() in lower_vocab:
                if term.lower() not in {b.lower() for b in basic_clean}:
                    basic_clean.append(term)
            else:
                rebalanced_adv.append(term)
        adv_clean = rebalanced_adv

    # required_terms는 기존 Tab 2 매칭 호환을 위해 상위·기본 학술 용어 전체 합집합으로 유지
    merged_req = _clean_term_list([*adv_clean, *basic_clean, *req_clean])
    if not adv_clean and not basic_clean and merged_req:
        adv_clean = list(merged_req)

    extracted.advanced_terms = adv_clean
    extracted.basic_terms = basic_clean
    extracted.required_terms = merged_req
    extracted.needs_review = needs_review
    extracted.review_notes = list(dict.fromkeys(review_notes))
    return extracted
