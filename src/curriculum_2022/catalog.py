from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "curriculum_2022.json"
_MOE_SEED_PATH = Path(__file__).resolve().parents[2] / "data" / "curriculum_2022_moe_seed.json"
_RUNTIME_STANDARDS_PATH = Path(__file__).resolve().parents[2] / "exam_db" / "ncic_standards.json"
_CATALOG_CACHE: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}


def _stat_sig(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
        return (int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        return None
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


def _load_curriculum_2022_catalog_readonly(path: Path | None = None) -> dict[str, Any]:
    """2022 개정 교육과정 공식 편제 및 NCIC 누적 성취기준 JSON을 읽기 전용 캐시 참조로 반환한다."""
    target = path or _DATA_PATH
    if path is None:
        cache_key = f"default::{target}::{_RUNTIME_STANDARDS_PATH}"
        sig = (_stat_sig(target), _stat_sig(_RUNTIME_STANDARDS_PATH))
    else:
        cache_key = f"custom::{target}"
        sig = (_stat_sig(target),)

    cached = _CATALOG_CACHE.get(cache_key)
    if cached is not None and cached[0] == sig:
        return cached[1]

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

    _CATALOG_CACHE[cache_key] = (sig, catalog)
    return catalog


def load_curriculum_2022_catalog(path: Path | None = None) -> dict[str, Any]:
    """2022 개정 교육과정 공식 편제 및 NCIC 누적 성취기준 JSON을 로드한다 (mtime 기반 메모리 캐시 적용)."""
    return copy.deepcopy(_load_curriculum_2022_catalog_readonly(path))


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
    cat_data = (catalog or _load_curriculum_2022_catalog_readonly()).get("categories", {})
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
    cat_data = (catalog or _load_curriculum_2022_catalog_readonly()).get("categories", {})
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
    cat_map = (catalog or _load_curriculum_2022_catalog_readonly()).get("categories", {})
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
    cat_obj = catalog or _load_curriculum_2022_catalog_readonly()
    entry = cat_obj.get("categories", {}).get(subject_category, {})
    return bool(entry.get("standards"))
