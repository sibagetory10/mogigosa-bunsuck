from __future__ import annotations

from datetime import datetime

from google.genai import types

from src.gemini.core import (
    _build_client,
    _generate_content_with_fallback,
    _generate_structured_with_fallback,
    _parse_structured,
)
from src.models import EnglishDictEntry, normalize_english_dict_key


def lookup_english_word_with_gemini(
    api_key: str,
    model_name: str,
    query: str,
    context_sentences: list[str] | None = None,
) -> EnglishDictEntry:
    """수능·모의고사 영어 독해 기준으로 영단어·숙어(또는 한국어→영어)를 정밀 조회해 구조화 엔트리를 반환한다."""
    clean_query = (query or "").strip()
    if not clean_query:
        raise ValueError("검색할 영단어나 숙어(또는 한국어 뜻)를 입력해 주세요.")

    client = _build_client(api_key, timeout_seconds=45)
    system_inst = (
        "당신은 대한민국 수능·평가원·모의고사 영어 독해 및 어휘 출제 전문 사전 편찬위원입니다.\n"
        "사용자가 입력한 영단어, 숙어/구동사, 활용형(예: resilient, took into account), "
        "또는 한국어 뜻(예: '회복탄력성')을 바탕으로 수능·고교 영어 학습에 최적화된 정밀 사전 항목을 JSON(EnglishDictEntry)으로 작성하세요.\n\n"
        "[작성 원칙]\n"
        "1. `headword`: 표제어 기본 원형(단어는 기본형, 구동사/숙어는 대표 원형, 한국어 입력 시 가장 적절한 대표 수능 영단어 1개)으로 정규화하세요.\n"
        "2. `phonetic`: 미국식 표준 IPA 발음 기호를 `[...]` 형식으로 표기하세요.\n"
        "3. `cefr_level`: 반드시 `'필수 기초'`, `'고1~2 핵심'`, `'수능 1등급 필수'`, `'고난도 심화'` 중 하나를 선택하세요.\n"
        "4. `summary_ko`: 전체 의미를 대표하는 간결하고 정확한 한국어 뜻 요약 1줄.\n"
        "5. `senses` (2~4개): 수능·모의고사 지문에서 실제 자주 출제되는 품사·의미를 우선 배치하세요. "
        "각 항목마다 `pos`(한국어 품사명), `meaning_ko`, `nuance_note`(다의어/혼동 어휘 구별 포인트), "
        "`example_en`(수능 독해 톤의 자연스러운 영어 예문 1문장), `example_ko`(자연스러운 한국어 번역)를 채우세요.\n"
        "6. `synonyms` (3~5개): 수능 빈칸/어휘/주제 선택지 패러프레이징에 자주 짝지어 나오는 유의어를 `'word (뜻)'` 형식으로.\n"
        "7. `antonyms` (1~3개): `'word (뜻)'` 형식 (없으면 빈 리스트).\n"
        "8. `derivatives` (2~4개): `'word (품사: 뜻)'` 형식의 핵심 파생어.\n"
        "9. `collocations` (2~4개): `'phrase (뜻)'` 형식의 빈출 연어/숙어 결합.\n"
        "10. `etymology_tip`: 접두사·어근·어원 스토리 기반 기억법 1~2문장.\n"
        "11. `csat_tip`: 수능·평가원 시험에서 자주 나오는 2차 의미(다의어 함정), 전치사 호응, 패러프레이징 출제 포인트 1~2문장."
    )

    ctx_lines = [s.strip() for s in (context_sentences or []) if isinstance(s, str) and s.strip()][:2]
    ctx_block = ""
    if ctx_lines:
        formatted_ctx = "\n".join(f"- 예문 {i}: {s}" for i, s in enumerate(ctx_lines, start=1))
        ctx_block = (
            f"\n\n[저장된 모의고사 기출 지문 내 실제 등장 문맥]\n{formatted_ctx}\n"
            "(위 기출 문맥에서 쓰인 품사와 의미가 `senses`에 반드시 포함되도록 해 주세요.)"
        )

    prompt = f"조회 대상: `{clean_query}`{ctx_block}\n\n위 어휘의 수능·고교 맞춤형 정밀 사전 데이터를 JSON으로 생성하세요."

    entry = _generate_structured_with_fallback(
        client=client,
        model_name=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=EnglishDictEntry,
            temperature=0.1,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL,
                include_thoughts=False,
            ),
        ),
        fast_mode=True,
        schema_cls=EnglishDictEntry,
        max_parse_retries=2,
    )
    head = (entry.headword or clean_query).strip()
    return entry.model_copy(
        update={
            "headword": head,
            "query_normalized": normalize_english_dict_key(head),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
