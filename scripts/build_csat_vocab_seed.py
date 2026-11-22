from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import threading
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FLASH_LITE_MODEL, resolve_credentials  # noqa: E402
from src.models import (  # noqa: E402
    EnglishDictEntry,
    EnglishDictSense,
    normalize_english_dict_key,
)

DEFAULT_OUTPUT_PATH = ROOT / "data" / "csat_vocab_seed.json"
DEFAULT_TARGET_COUNT = 3000

_HEADWORD_BUCKETS: tuple[str, ...] = (
    "교육부 고등 영어 필수 기초 동사 (알파벳 A~L 시작 위주)",
    "교육부 고등 영어 필수 기초 동사 (알파벳 M~Z 시작 위주)",
    "교육부 고등 영어 필수 핵심 명사 (알파벳 A~L 시작 위주)",
    "교육부 고등 영어 필수 핵심 명사 (알파벳 M~Z 시작 위주)",
    "교육부 고등 영어 필수 핵심 형용사·부사 (알파벳 A~L 시작 위주)",
    "교육부 고등 영어 필수 핵심 형용사·부사 (알파벳 M~Z 시작 위주)",
    "수능·평가원 모의고사 빈출 철학·심리·인지과학·인식론 어휘",
    "수능·평가원 모의고사 빈출 사회·문화·인류학·역사 어휘",
    "수능·평가원 모의고사 빈출 경제·경영·시장·소비·금융 어휘",
    "수능·평가원 모의고사 빈출 정치·법률·제도·윤리·공공정책 어휘",
    "수능·평가원 모의고사 빈출 생명과학·진화·생태계·유전 어휘",
    "수능·평가원 모의고사 빈출 물리·화학·지구과학·환경·기후 어휘",
    "수능·평가원 모의고사 빈출 과학기술·AI·정보통신·미디어 어휘",
    "수능·평가원 모의고사 빈출 예술·문학·미학·음악·건축 어휘",
    "수능 독해 논리·인과·비교·대조·추론·증명 관련 핵심 어휘",
    "수능 독해 변화·증감·유지·붕괴·회복·확장 관련 핵심 어휘",
    "수능 독해 인간관계·감정·태도·성향·동기부여 관련 핵심 어휘",
    "수능·평가원 빈출 의학·건강·영양·질병·뇌과학 어휘",
    "수능·평가원 빈출 교육·학습·언어학·소통·설득 어휘",
    "수능 1등급 고난도 심화 동사·형용사 (알파벳 A~L 시작)",
    "수능 1등급 고난도 심화 동사·형용사 (알파벳 M~Z 시작)",
    "수능 1등급 고난도 심화 추상 명사 및 학술 어휘",
    "수능·평가원 빈출 다의어 및 형태 유사 혼동 어휘 핵심",
    "수능 독해 필수 구동사·전치사구·접속부사·핵심 숙어 (2~3단어 숙어 위주)",
    "알파벳 A~B 고교·수능 필수 어휘 보충",
    "알파벳 C~D 고교·수능 필수 어휘 보충",
    "알파벳 E~H 고교·수능 필수 어휘 보충",
    "알파벳 I~M 고교·수능 필수 어휘 보충",
    "알파벳 N~P 고교·수능 필수 어휘 보충",
    "알파벳 R~S 고교·수능 필수 어휘 보충",
    "알파벳 T~Z 고교·수능 필수 어휘 보충",
    "EBS 수능특강·수능완성 및 최근 5개년 평가원 고빈도 핵심 어휘",
)


_ALPHA_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("알파벳 A로 시작하는 수능·평가원·EBS 필수 어휘 (아직 등록되지 않은 단어만)", ("a",)),
    ("알파벳 B로 시작하는 수능·평가원·EBS 필수 어휘 (아직 등록되지 않은 단어만)", ("b",)),
    ("알파벳 C로 시작하는 수능·평가원·EBS 필수 어휘 (아직 등록되지 않은 단어만)", ("c",)),
    ("알파벳 D로 시작하는 수능·평가원·EBS 필수 어휘 (아직 등록되지 않은 단어만)", ("d",)),
    ("알파벳 E로 시작하는 수능·평가원·EBS 필수 어휘 (아직 등록되지 않은 단어만)", ("e",)),
    ("알파벳 F~G로 시작하는 수능·평가원·EBS 필수 어휘 (아직 등록되지 않은 단어만)", ("f", "g")),
    ("알파벳 H~I로 시작하는 수능·평가원·EBS 필수 어휘 (아직 등록되지 않은 단어만)", ("h", "i")),
    ("알파벳 J~L로 시작하는 수능·평가원·EBS 필수 어휘 (아직 등록되지 않은 단어만)", ("j", "k", "l")),
    ("알파벳 M~N으로 시작하는 수능·평가원·EBS 필수 어휘 (아직 등록되지 않은 단어만)", ("m", "n")),
    ("알파벳 O~P로 시작하는 수능·평가원·EBS 필수 어휘 (아직 등록되지 않은 단어만)", ("o", "p")),
    ("알파벳 Q~R로 시작하는 수능·평가원·EBS 필수 어휘 (아직 등록되지 않은 단어만)", ("q", "r")),
    ("알파벳 S로 시작하는 수능·평가원·EBS 필수 어휘 (아직 등록되지 않은 단어만)", ("s",)),
    ("알파벳 T~U로 시작하는 수능·평가원·EBS 필수 어휘 (아직 등록되지 않은 단어만)", ("t", "u")),
    ("알파벳 V~Z로 시작하는 수능·평가원·EBS 필수 어휘 (아직 등록되지 않은 단어만)", ("v", "w", "x", "y", "z")),
    ("수능 독해 필수 구동사·전치사구·숙어 (2~3단어 숙어, 아직 등록되지 않은 표현만)", ()),
)


def _log(msg: str) -> None:
    print(msg, flush=True)


class HeadwordListOutput(BaseModel):
    words: list[str] = Field(default_factory=list)


class SeedSense(BaseModel):
    pos: str = Field(..., description="한국어 품사명 (예: 동사, 명사, 형용사, 부사, 숙어)")
    meaning_ko: str = Field(..., description="해당 품사의 핵심 한국어 뜻 (쉼표 구분 2~4개)")
    nuance_note: str = Field(default="", description="수능 독해 문맥 뉘앙스 또는 혼동 주의점 1줄")
    example_en: str = Field(..., description="수능·모의고사 독해 수준의 영문 예문 1문장")
    example_ko: str = Field(..., description="위 영문 예문의 자연스러운 한국어 해석")


class SeedDictItem(BaseModel):
    headword: str = Field(..., description="표제어 원형")
    phonetic: str = Field(default="", description="미국식 IPA 발음 기호 ([...])")
    cefr_level: Literal["필수 기초", "고1~2 핵심", "수능 1등급 필수", "고난도 심화"]
    summary_ko: str = Field(..., description="표제어 핵심 의미 1줄 요약")
    senses: list[SeedSense] = Field(default_factory=list, description="주요 품사·의미 1~2개")
    synonyms: list[str] = Field(default_factory=list, description="핵심 수능 유의어 2~3개 ('word (뜻)')")
    csat_tip: str = Field(default="", description="수능·평가원 출제 포인트 1문장")


class SeedBatchOutput(BaseModel):
    items: list[SeedDictItem] = Field(default_factory=list)


def _load_dotenv_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip().strip('"').strip("'")
        if key and val:
            os.environ.setdefault(key, val)


def resolve_script_gemini_api_key() -> str:
    """로컬 .env -> 환경변수 -> .streamlit/secrets.toml -> resolve_credentials 순으로 API 키를 로드한다."""
    _load_dotenv_file(ROOT / ".env")
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key

    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            data = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
            for key_name in ("GEMINI_API_KEY", "gemini_api_key"):
                val = data.get(key_name)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        except Exception:
            pass

    creds = resolve_credentials()
    if creds.gemini_api_key:
        return creds.gemini_api_key

    raise RuntimeError(
        "GEMINI_API_KEY를 찾을 수 없습니다. 프로젝트 루트의 .env 또는 .streamlit/secrets.toml에 키를 설정해 주세요."
    )


def _build_client(api_key: str, timeout_seconds: int = 60) -> genai.Client:
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=timeout_seconds * 1000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def _call_gemini_structured(
    client: genai.Client,
    *,
    system_instruction: str,
    prompt: str,
    schema: type[BaseModel],
    max_retries: int = 4,
) -> Any:
    delays = (2.0, 5.0, 10.0, 20.0)
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=FLASH_LITE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.2,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.MINIMAL,
                        include_thoughts=False,
                    ),
                ),
            )
            if getattr(resp, "parsed", None) is not None:
                return resp.parsed
            raw_text = (getattr(resp, "text", "") or "").strip()
            if not raw_text:
                raise ValueError("Gemini 응답이 비어 있습니다.")
            return schema.model_validate_json(raw_text)
        except Exception as exc:
            last_err = exc
            msg = str(exc).lower()
            fatal = any(
                k in msg
                for k in (
                    "401",
                    "403",
                    "api key not valid",
                    "api_key_invalid",
                    "permission_denied",
                    "unauthenticated",
                )
            )
            if fatal:
                raise
            if attempt < max_retries:
                wait_s = delays[min(attempt, len(delays) - 1)] + random.uniform(0.2, 1.0)
                time.sleep(wait_s)
                continue
            raise
    assert last_err is not None
    raise last_err


def _is_valid_english_term(word: str) -> bool:
    clean = normalize_english_dict_key(word)
    if not clean or len(clean) < 2 or len(clean) > 48:
        return False
    return bool(re.fullmatch(r"[a-z][a-z'\-\s]{1,47}", clean))


def collect_headwords_for_bucket(
    client: genai.Client,
    bucket_desc: str,
    count_per_bucket: int = 120,
    avoid_sample: list[str] | None = None,
) -> list[str]:
    sys_inst = (
        "당신은 대한민국 교육부 2022 개정 영어과 교육과정 및 수능·평가원 영어 어휘 편찬위원입니다.\n"
        "요청받은 카테고리에 해당하는 고등·수능 필수 영단어 또는 핵심 숙어 표제어(기본 원형)만 중복 없이 정확히 출력하세요.\n"
        "철자가 정확하고 실제 수능·모의고사 지문에 자주 등장하는 어휘만 포함하세요."
    )
    avoid_block = ""
    if avoid_sample:
        avoid_block = "\n이미 등록된 제외 목록(절대 중복 출력 금지): " + ", ".join(avoid_sample[:400])
    prompt = (
        f"카테고리: {bucket_desc}\n"
        f"목표 수량: {count_per_bucket}개\n"
        "위 카테고리에 해당하는 수능·고교 필수 표제어(단어는 기본 원형, 숙어는 기본형) 목록을 JSON으로 반환하세요."
        f"{avoid_block}"
    )
    parsed: HeadwordListOutput = _call_gemini_structured(
        client,
        system_instruction=sys_inst,
        prompt=prompt,
        schema=HeadwordListOutput,
    )
    out: list[str] = []
    seen: set[str] = set()
    for w in parsed.words:
        norm = normalize_english_dict_key(w)
        if _is_valid_english_term(norm) and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def generate_entries_batch(
    client: genai.Client,
    words: list[str],
) -> list[EnglishDictEntry]:
    sys_inst = (
        "당신은 대한민국 수능·평가원 영어 독해 및 어휘 전문 사전 편찬위원입니다.\n"
        "주어진 영단어·숙어 목록 각각에 대해 수능·고교 학습용 핵심 완비형 사전 엔트리를 JSON으로 생성하세요.\n"
        "[작성 규칙]\n"
        "1. 입력된 단어 하나당 items 배열에 정확히 1개 엔트리를 빠짐없이 작성하세요.\n"
        "2. `headword`: 입력 단어의 정확한 표제어 원형.\n"
        "3. `phonetic`: 미국식 표준 IPA 발음 기호 `[...]` 형식.\n"
        "4. `cefr_level`: 반드시 `'필수 기초'`, `'고1~2 핵심'`, `'수능 1등급 필수'`, `'고난도 심화'` 중 하나.\n"
        "5. `summary_ko`: 대표 한국어 뜻 1줄 요약.\n"
        "6. `senses`: 가장 중요한 1~2개 품사·의미만 간결하고 정확하게 작성. 각 항목마다 `pos`, `meaning_ko`, `nuance_note`(짧은 1줄), `example_en`(수능 독해 톤 1문장), `example_ko`(한국어 번역) 필수.\n"
        "7. `synonyms`: 수능 빈출 유의어 2~3개 (`'word (뜻)'` 형식).\n"
        "8. `csat_tip`: 수능 출제 포인트 또는 다의어/파생 주의점 1문장."
    )
    word_lines = "\n".join(f"- {w}" for w in words)
    prompt = f"다음 {len(words)}개 영단어·숙어 각각의 사전 데이터를 빠짐없이 생성하세요:\n{word_lines}"
    parsed: SeedBatchOutput = _call_gemini_structured(
        client,
        system_instruction=sys_inst,
        prompt=prompt,
        schema=SeedBatchOutput,
    )
    now_str = datetime.now().isoformat(timespec="seconds")
    results: list[EnglishDictEntry] = []
    for item in parsed.items:
        head = (item.headword or "").strip()
        norm = normalize_english_dict_key(head)
        if not _is_valid_english_term(norm):
            continue
        senses = [
            EnglishDictSense(
                pos=s.pos.strip(),
                meaning_ko=s.meaning_ko.strip(),
                nuance_note=s.nuance_note.strip(),
                example_en=s.example_en.strip(),
                example_ko=s.example_ko.strip(),
            )
            for s in item.senses[:2]
            if s.pos.strip() and s.meaning_ko.strip()
        ]
        if not senses and item.summary_ko.strip():
            senses = [
                EnglishDictSense(
                    pos="핵심",
                    meaning_ko=item.summary_ko.strip(),
                )
            ]
        entry = EnglishDictEntry(
            headword=head,
            query_normalized=norm,
            phonetic=item.phonetic.strip(),
            cefr_level=item.cefr_level,
            summary_ko=item.summary_ko.strip(),
            senses=senses,
            synonyms=[s.strip() for s in item.synonyms[:3] if s.strip()],
            csat_tip=item.csat_tip.strip(),
            updated_at=now_str,
        )
        results.append(entry)
    return results


def _atomic_save_seed(path: Path, entries: dict[str, dict[str, Any]], aliases: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = {"entries": entries, "aliases": aliases}
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _load_existing_seed(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    if not path.exists():
        return {}, {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}, {}
        entries = {
            str(k): dict(v)
            for k, v in (data.get("entries") or {}).items()
            if isinstance(v, dict)
        }
        aliases = {
            str(k): str(v)
            for k, v in (data.get("aliases") or {}).items()
            if isinstance(v, str)
        }
        return entries, aliases
    except Exception:
        return {}, {}


def _collect_alpha_candidates(
    client: genai.Client,
    seen_candidates: set[str],
    candidate_pool: list[str],
    workers: int,
    count_per_bucket: int = 100,
) -> int:
    """알파벳 구간별로 이미 등록된 단어를 제외 목록으로 전달해 신규 수능 표제어를 수집한다."""
    total_added = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {}
        for desc, prefixes in _ALPHA_BUCKETS:
            if prefixes:
                avoid = [w for w in seen_candidates if w and w[0] in prefixes]
            else:
                avoid = [w for w in seen_candidates if " " in w]
            fut = pool.submit(
                collect_headwords_for_bucket,
                client,
                desc,
                count_per_bucket,
                avoid[:400],
            )
            future_map[fut] = desc

        for fut in concurrent.futures.as_completed(future_map):
            desc = future_map[fut]
            try:
                words = fut.result()
                added_here = 0
                for w in words:
                    if w not in seen_candidates:
                        seen_candidates.add(w)
                        candidate_pool.append(w)
                        added_here += 1
                        total_added += 1
                _log(f"  - 알파벳 버킷 완료 (+{added_here}개, 신규 후보 {len(candidate_pool)}개): {desc[:28]}...")
            except Exception as exc:
                _log(f"  [경고] 알파벳 버킷 실패 ({desc[:25]}): {exc}")
    return total_added


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemini API로 수능 필수 어휘 시드 JSON을 생성합니다.")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET_COUNT, help="목표 고유 표제어 수 (기본 3000)")
    parser.add_argument("--batch-size", type=int, default=25, help="API 1회 호출당 생성 단어 수 (기본 25)")
    parser.add_argument("--workers", type=int, default=8, help="병렬 스레드 수 (기본 8)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="출력 JSON 경로")
    args = parser.parse_args()

    api_key = resolve_script_gemini_api_key()
    client = _build_client(api_key, timeout_seconds=75)

    entries, aliases = _load_existing_seed(args.output)
    _log(f"[시드 로더] 기존 저장 엔트리: {len(entries)}개 (목표: {args.target}개)")
    if len(entries) >= args.target:
        _log(f"[완료] 이미 목표 수량({args.target}개) 이상이 존재합니다: {len(entries)}개")
        return 0

    lock = threading.Lock()
    candidate_pool: list[str] = []
    seen_candidates: set[str] = set(entries.keys())

    # 1단계: 기존 단어 수에 따라 초기 버킷 또는 알파벳별 중복 회피 버킷 수집
    need_candidates = max(0, (args.target - len(entries)) + 200)
    _log(f"[1단계] 수능 필수 표제어 수집 시작 (필요 신규 후보 약 {need_candidates}개)...")

    if len(entries) == 0:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_map = {
                pool.submit(collect_headwords_for_bucket, client, desc, 125): desc
                for desc in _HEADWORD_BUCKETS
            }
            for fut in concurrent.futures.as_completed(future_map):
                desc = future_map[fut]
                try:
                    words = fut.result()
                    added_here = 0
                    for w in words:
                        if w not in seen_candidates:
                            seen_candidates.add(w)
                            candidate_pool.append(w)
                            added_here += 1
                    _log(f"  - 버킷 완료 (+{added_here}개, 누적 후보 {len(candidate_pool)}개): {desc[:28]}...")
                except Exception as exc:
                    _log(f"  [경고] 버킷 실패 ({desc[:25]}): {exc}")

    # 목표 대비 부족 시 알파벳별 제외 목록 기반 고속 수집
    alpha_round = 0
    while len(candidate_pool) < need_candidates and alpha_round < 5:
        alpha_round += 1
        _log(f"[1단계-보충 {alpha_round}차] 알파벳 구간별 기존 단어 제외 수집 시작...")
        added_alpha = _collect_alpha_candidates(
            client,
            seen_candidates,
            candidate_pool,
            workers=args.workers,
            count_per_bucket=90,
        )
        _log(f"[1단계-보충 {alpha_round}차 완료] +{added_alpha}개 (신규 후보 총 {len(candidate_pool)}개)")
        if added_alpha == 0:
            break

    # 2단계: 배치 단위 사전 데이터 생성 및 체크포인트 저장
    round_idx = 0
    while len(entries) < args.target and round_idx < 5:
        round_idx += 1
        remaining_needed = args.target - len(entries)
        target_batch_words = [w for w in candidate_pool if w not in entries]
        if len(target_batch_words) < remaining_needed:
            _log(f"[2단계 전 보충] 후보 부족({len(target_batch_words)} < {remaining_needed}), 알파벳 버킷 추가 수집...")
            _collect_alpha_candidates(
                client,
                seen_candidates,
                candidate_pool,
                workers=args.workers,
                count_per_bucket=90,
            )
            target_batch_words = [w for w in candidate_pool if w not in entries]

        if not target_batch_words:
            break

        slice_limit = remaining_needed + min(120, max(25, remaining_needed // 10))
        to_process = target_batch_words[:slice_limit]
        batches = [
            to_process[i : i + args.batch_size]
            for i in range(0, len(to_process), args.batch_size)
        ]
        _log(
            f"[2단계 라운드 {round_idx}] 남은 목표 {remaining_needed}개 -> {len(to_process)}개 단어 "
            f"({len(batches)}배치, 동시 {args.workers}스레드) 생성 시작..."
        )

        completed_batches = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            fut_map = {
                pool.submit(generate_entries_batch, client, b): b
                for b in batches
            }
            for fut in concurrent.futures.as_completed(fut_map):
                completed_batches += 1
                try:
                    batch_entries = fut.result()
                except Exception as exc:
                    _log(f"  [경고] 배치 {completed_batches}/{len(batches)} 실패: {exc}")
                    continue

                with lock:
                    for ent in batch_entries:
                        nk = ent.query_normalized or normalize_english_dict_key(ent.headword)
                        if nk and nk not in entries:
                            entries[nk] = ent.model_dump()
                    if completed_batches % 3 == 0 or completed_batches == len(batches) or len(entries) >= args.target:
                        _atomic_save_seed(args.output, entries, aliases)
                _log(
                    f"  진행률: {completed_batches}/{len(batches)} 배치 완료 | "
                    f"누적 저장 단어: {len(entries)} / {args.target}개"
                )

    _atomic_save_seed(args.output, entries, aliases)
    size_kb = args.output.stat().st_size // 1024 if args.output.exists() else 0
    _log(f"[완료] {args.output} 저장 완료: 총 {len(entries)}개 단어 ({size_kb} KB)")
    return 0 if len(entries) >= args.target else 1


if __name__ == "__main__":
    raise SystemExit(main())
