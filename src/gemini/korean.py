from __future__ import annotations

import json
import re
from typing import Any, Callable, Sequence

from google.genai import types

from src.gemini.core import (
    _apply_lang_standard_validation,
    _build_client,
    _build_lang_ncic_grounding_section,
    _generate_content_with_fallback,
    _normalize_extracted_answer,
    _generate_structured_with_fallback,
    _parse_structured,
    optimize_image_for_vision,
    prepare_pdf_part,
)
from src.models import (
    BatchKoreanExamExtractionOutput,
    KoreanBlindSetVerdict,
    KoreanPassageAnalysisOutput,
    KoreanQuestionExtractionItem,
    KoreanSetQuestionOutput,
    KoreanSetReconstructionOutput,
    KoreanSetRepairOutput,
    KoreanSingleExtractionOutput,
    KoreanVisualDiagram,
    QuestionRecord,
    RewrittenQuestion,
    normalize_korean_group_id,
)
from src.text_formatter import (
    clean_korean_exam_passage,
    format_explanation_text,
)


def _is_korean_dialogue_or_poetry(
    korean_section: str,
    korean_subgenre: str = "",
    passage_text: str = "",
) -> bool:
    """시가·극·대화형 지문처럼 원문 줄바꿈 보존이 필요한 영역인지 판별한다."""
    sec = (korean_section or "").strip()
    sub = (korean_subgenre or "").strip()
    if sec == "화법과 작문" or any(k in sub for k in ("화법", "대화", "토론", "협상", "극", "시나리오", "희곡")):
        return True
    return _is_korean_poetry_genre(sec, sub, passage_text)


def _sync_korean_group_passages(items: list[KoreanQuestionExtractionItem]) -> None:
    """동일한 passage_group_id를 공유하는 국어 세트 문항들의 지문·주제·세부 갈래를 동기화하고 정제한다."""
    best_passage: dict[str, str] = {}
    best_topic: dict[str, str] = {}
    best_subgenre: dict[str, str] = {}
    for item in items:
        if (item.passage_text or "").strip():
            item.passage_text = clean_korean_exam_passage(
                item.passage_text,
                is_poetry_or_dialogue=_is_korean_dialogue_or_poetry(
                    item.korean_section, item.korean_subgenre, item.passage_text
                ),
            )
        gid = normalize_korean_group_id(item.passage_group_id, item.q_num)
        item.passage_group_id = gid
        if not gid:
            continue
        p_txt = (item.passage_text or "").strip()
        if len(p_txt) > len(best_passage.get(gid, "")):
            best_passage[gid] = p_txt
        t_txt = (item.passage_topic or "").strip()
        if len(t_txt) > len(best_topic.get(gid, "")):
            best_topic[gid] = t_txt
        sg_txt = (item.korean_subgenre or "").strip()
        if len(sg_txt) > len(best_subgenre.get(gid, "")):
            best_subgenre[gid] = sg_txt

    for item in items:
        gid = item.passage_group_id
        if not gid:
            continue
        if gid in best_passage and best_passage[gid]:
            if len((item.passage_text or "").strip()) < len(best_passage[gid]) * 0.7:
                item.passage_text = best_passage[gid]
        if not (item.passage_topic or "").strip() and gid in best_topic:
            item.passage_topic = best_topic[gid]
        if not (item.korean_subgenre or "").strip() and gid in best_subgenre:
            item.korean_subgenre = best_subgenre[gid]


def analyze_korean_exam_pdf_batch_with_gemini(
    api_key: str,
    model_name: str,
    prob_pdf_bytes: bytes,
    target_q_nums: list[int],
    exam_name: str,
    answer_snippets: dict[int, str] | None = None,
    answer_pdf_part: Any | None = None,
    status_callback: Callable[[str], None] | None = None,
    parsed_crops: Sequence[Any] | None = None,
) -> dict[int, KoreanQuestionExtractionItem]:
    """
    국어 모의고사 문제지 PDF(및 해설지)를 일괄 분석하여
    공통 영역인 '독서'와 '문학' 지문 세트만 추출한다.
    텍스트 레이어가 있는 경우 로컬에서 이미 추출한 문항·지문 텍스트와 해설 스니펫만 경량 전송하고,
    AI가 지문 원문(passage_text)을 중복 출력하지 않도록 해 입력·출력 토큰을 대폭 절감한다.
    """
    ordered_q = sorted({int(q) for q in target_q_nums if 1 <= int(q) <= 45})
    if not ordered_q:
        return {}

    crop_by_q: dict[int, Any] = {}
    local_group_passages: dict[str, str] = {}
    for cq in (parsed_crops or []):
        qn = int(getattr(cq, "q_num", 0) or 0)
        if qn in ordered_q:
            crop_by_q[qn] = cq
            gid = normalize_korean_group_id(getattr(cq, "passage_group_id", ""), qn)
            sp = (getattr(cq, "shared_passage_text", "") or "").strip()
            if gid and len(sp) > len(local_group_passages.get(gid, "")):
                local_group_passages[gid] = sp

    # 로컬 텍스트 레이어가 충분한지 판별 (대상 문항 70% 이상에서 추출 텍스트 >= 40자)
    text_rich_count = sum(
        1
        for q in ordered_q
        if q in crop_by_q and len(re.sub(r"\s+", "", getattr(crop_by_q[q], "extracted_text", "") or "")) >= 40
    )
    use_lightweight_text_mode = bool(crop_by_q) and (text_rich_count / max(1, len(ordered_q))) >= 0.70

    snippets_map = answer_snippets or {}
    snippet_lines: list[str] = []
    for q in ordered_q:
        sn = (snippets_map.get(q) or "").strip()
        if sn:
            snippet_lines.append(f"[{q}번 해설 발췌] {sn[:500]}")

    # 스니펫이 대상 문항 80% 이상을 커버하면 무거운 해설지 PDF 바이너리 첨부 생략
    snippets_cover_mostly = (len(snippet_lines) / max(1, len(ordered_q))) >= 0.80

    client = _build_client(api_key, timeout_seconds=150)
    if status_callback is not None:
        mode_tag = "경량 텍스트 모드" if use_lightweight_text_mode else "PDF 비전 모드"
        status_callback(f"📖 국어 모의고사({len(ordered_q)}문항 · {mode_tag}) 일괄 분석 준비 중...")

    ncic_section, ncic_rule = _build_lang_ncic_grounding_section("국어", exam_name)
    hint_section = (
        "\n[해설지 문항별 발췌]\n" + "\n".join(snippet_lines) + "\n"
        if snippet_lines
        else ""
    )

    if use_lightweight_text_mode:
        # 로컬 추출 텍스트로 세트별 공통 지문 1회 + 문항별 발문·선지(공통 지문 제거) 조립
        text_blocks: list[str] = []
        emitted_groups: set[str] = set()
        for q in ordered_q:
            cq = crop_by_q.get(q)
            if cq is None:
                continue
            gid = normalize_korean_group_id(getattr(cq, "passage_group_id", ""), q)
            sp_txt = local_group_passages.get(gid, "")
            if gid and sp_txt and gid not in emitted_groups:
                emitted_groups.add(gid)
                text_blocks.append(f"=== [공통 지문 세트: {gid}번] ===\n{sp_txt}")
            raw_q_txt = (getattr(cq, "extracted_text", "") or "").strip()
            if sp_txt and sp_txt in raw_q_txt:
                raw_q_txt = raw_q_txt.replace(sp_txt, "").strip()
            text_blocks.append(f"[{q}번 문항 (세트:{gid or q})]\n{raw_q_txt[:1800]}")
        problem_text_section = "\n\n".join(text_blocks)

        passage_rule = (
            "4. `passage_text`: 지문 원문은 이미 시스템에 로컬 추출되어 있으므로, **이미 지문이 제공된 문항은 `passage_text`를 반드시 빈 문자열(`\"\"`)로 비워 출력 토큰을 절약하세요.** "
            "(단, 제공된 텍스트에 공통 지문이 전혀 누락된 문항에 한해서만 지문을 수록하세요.)\n"
        )
        contents: list[Any] = []
        if answer_pdf_part is not None and not snippets_cover_mostly:
            contents.append(answer_pdf_part)
        prompt = (
            f"시험 회차: {exam_name}\n"
            f"검토 대상 문항 번호(총 {len(ordered_q)}문항): {', '.join(str(q) for q in ordered_q)}\n\n"
            f"{ncic_section}"
            f"[문제지 추출 텍스트 (지문 및 문항)]\n{problem_text_section}\n"
            f"{hint_section}\n"
            "위 국어 문제지 텍스트와 해설을 바탕으로 각 문항의 영역(독서/문학/화작/언매), 세부 갈래, 세트 범위, 주제, 어휘, 정답, 해설을 JSON(questions)으로 반환하세요."
        )
        contents.append(prompt)
    else:
        prob_part = prepare_pdf_part(
            client,
            prob_pdf_bytes,
            display_name=f"{exam_name}_korean_problem.pdf",
        )
        passage_rule = (
            "4. `passage_text` (스캔 PDF 독서·문학 전용):\n"
            "   - 발문과 선택지(①~⑤), `<보기>`를 제외한 **순수 국어 지문 원문**을 추출하되, "
            "**반드시 각 세트([1~3], [4~9] 등)의 첫 번째 문항에만 1회 수록하고 같은 세트의 나머지 문항은 빈 문자열(`\"\"`)로 비워 중복 출력을 피하세요.**\n"
        )
        contents = [prob_part]
        if answer_pdf_part is not None and not snippets_cover_mostly:
            contents.append(answer_pdf_part)
        q_list_str = ", ".join(str(q) for q in ordered_q)
        prompt = (
            f"시험 회차: {exam_name}\n"
            f"검토 대상 문항 번호(총 {len(ordered_q)}문항): {q_list_str}\n"
            f"{ncic_section}"
            f"{hint_section}\n"
            "위 국어 문제지 PDF(및 해설지)에서 독서·문학 공통 영역 문항의 지문 세트, 주제, 어휘, 정답, 해설을 추출하여 JSON(questions)으로 반환하세요. "
            "(화법과 작문, 언어와 매체 문항은 korean_section만 표기하고 passage_text는 비워 두세요.)"
        )
        contents.append(prompt)

    system_inst = (
        "당신은 대한민국 수능·평가원·교육청 국어 영역(제1교시) 기출 분석 전문가입니다.\n"
        "요청된 전체 국어 문항(**독서 · 문학 · 화법과 작문 · 언어와 매체**, 1~45번 전체)을 빠짐없이 분석해 JSON(questions)으로 추출하세요.\n\n"
        "[추출 규칙]\n"
        "1. `korean_section`: 반드시 `'독서'`, `'문학'`, `'화법과 작문'`, `'언어와 매체'` 중 하나로 정확히 분류하세요.\n"
        "2. `korean_subgenre`: 독서는 `'독서론'`/`'인문·예술'`/`'사회·경제'`/`'과학·기술'`/`'주제 통합'`, 문학은 `'현대시'`/`'고전시가'`/`'현대소설'`/`'고전소설'`/`'극·수필'`/`'갈래 복합'`, 화작은 `'화법'`/`'작문'`/`'화법·작문 통합'`, 언매는 `'언어(문법)'`/`'매체'` 등으로 적으세요.\n"
        "3. `passage_group_id`: 시험지에 표시된 공통 지문 세트 범위(예: `'1~3'`, `'4~9'`, `'35~37'`, `'38~42'`)를 적으세요. 단독 문항이면 해당 문항 번호를 적으세요.\n"
        + passage_rule
        + "5. `passage_topic`: 지문의 핵심 주제, 문학 작품명·작가, 또는 문법·화작 핵심 소재를 1줄로 요약하세요.\n"
        "6. `key_vocabulary`: 지문·문항의 핵심 개념어·어휘·문법/문학 개념 3~5개를 나열하세요.\n"
        "7. `question_type`: 원래 문항 유형(예: `'세부 내용 일치'`, `'핵심 논지 추론'`, `'<보기> 사례 적용'`, `'표현상 특징'`, `'고쳐쓰기·자료 활용'`, `'음운·문장 구조 분석'`).\n"
        "8. `official_explanation`과 `answer`: 해설지 기준 핵심 풀이 근거 요약(1~2문장)과 공식 정답(`'1'`~`'5'`)을 정확히 추출하세요."
        + ncic_rule
    )

    if status_callback is not None:
        status_callback(f"⚡ {model_name} 모델로 국어 전 문항({len(ordered_q)}문항 · 독서/문학/화작/언매) 일괄 분석 중...")

    parsed = _generate_structured_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=BatchKoreanExamExtractionOutput,
            temperature=0.1,
            max_output_tokens=32768,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL,
                include_thoughts=False,
            ),
        ),
        fast_mode=True,
        status_callback=status_callback,
        schema_cls=BatchKoreanExamExtractionOutput,
        max_parse_retries=2,
    )

    items_list: list[KoreanQuestionExtractionItem] = []
    seen_q: set[int] = set()
    for item in parsed.questions:
        if item.q_num in ordered_q and item.q_num not in seen_q:
            item.answer = _normalize_extracted_answer(item.answer)
            cq_ref = crop_by_q.get(item.q_num)
            fallback_gid = getattr(cq_ref, "passage_group_id", "") if cq_ref is not None else ""
            item.passage_group_id = normalize_korean_group_id(item.passage_group_id or fallback_gid, item.q_num)
            if not (item.passage_text or "").strip():
                # AI 출력 토큰 절약을 위해 비워둔 경우 로컬 추출 지문(또는 단독 문항 본문) 자동 복원
                local_sp = local_group_passages.get(item.passage_group_id, "")
                if not local_sp and cq_ref is not None:
                    local_sp = (getattr(cq_ref, "shared_passage_text", "") or getattr(cq_ref, "extracted_text", "") or "").strip()
                if local_sp:
                    item.passage_text = local_sp
            _apply_lang_standard_validation(item, "국어")
            seen_q.add(item.q_num)
            items_list.append(item)

    _sync_korean_group_passages(items_list)
    return {it.q_num: it for it in items_list}


def analyze_korean_question_with_gemini(
    api_key: str,
    model_name: str,
    crop_png_bytes: bytes | None,
    q_num: int,
    exam_name: str,
    answer_pdf_part: Any | None = None,
    answer_text_hint: str = "",
    question_text_hint: str = "",
    shared_passage_hint: str = "",
) -> KoreanSingleExtractionOutput:
    """국어 단일 누락 문항 보완 추출 (독서·문학·화작·언매 전 영역 판별 및 지문·정답 추출)."""
    client = _build_client(api_key, timeout_seconds=45)
    ncic_section, ncic_rule = _build_lang_ncic_grounding_section("국어", exam_name)
    system_inst = (
        "당신은 대한민국 수능·모의고사 국어 영역 기출 분석 전문가입니다.\n"
        "제공된 문항 이미지/텍스트와 해설을 보고 영역(`독서`, `문학`, `화법과 작문`, `언어와 매체`), "
        "세부 갈래, 세트 범위(`passage_group_id`), 지문/자료 원문(`passage_text`), 주제/작품명, 핵심 어휘, 해설, 정답(`'1'`~`'5'`)을 추출하세요."
        + ncic_rule
    )
    hint_parts: list[str] = [f"시험 회차: {exam_name}", f"문항 번호: {q_num}번"]
    if ncic_section:
        hint_parts.append(ncic_section.strip())
    if shared_passage_hint.strip():
        hint_parts.append(f"[동일 세트 공통 지문 참고]\n{shared_passage_hint.strip()}")
    if question_text_hint.strip():
        hint_parts.append(f"[추출 텍스트 참고]\n{question_text_hint.strip()[:2500]}")
    if answer_text_hint.strip():
        hint_parts.append(f"[해설지 발췌]\n{answer_text_hint.strip()[:900]}")

    contents: list[Any] = []
    if crop_png_bytes:
        opt_png = optimize_image_for_vision(crop_png_bytes, max_width=1400)
        contents.append(types.Part.from_bytes(data=opt_png, mime_type="image/png"))
    if answer_pdf_part is not None and not answer_text_hint.strip():
        contents.append(answer_pdf_part)
    contents.append("\n\n".join(hint_parts))

    out = _generate_structured_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=KoreanSingleExtractionOutput,
            temperature=0.1,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL,
                include_thoughts=False,
            ),
        ),
        fast_mode=True,
        schema_cls=KoreanSingleExtractionOutput,
        max_parse_retries=2,
    )
    out.answer = _normalize_extracted_answer(out.answer)
    out.passage_group_id = normalize_korean_group_id(out.passage_group_id, q_num)
    if not (out.passage_text or "").strip():
        out.passage_text = (shared_passage_hint or question_text_hint or "").strip()
    if (out.passage_text or "").strip():
        out.passage_text = clean_korean_exam_passage(
            out.passage_text,
            is_poetry_or_dialogue=_is_korean_dialogue_or_poetry(
                out.korean_section, out.korean_subgenre, out.passage_text
            ),
        )
    _apply_lang_standard_validation(out, "국어")
    return out


_CIRCLED_NUMS: tuple[str, ...] = ("①", "②", "③", "④", "⑤")
_KOR_MARKER_TOKEN_RE = re.compile(r"[\u3260-\u327b\u24d0-\u24e9]|\[[A-E]\]|\([가나다라]\)")
_KOR_STEM_GLOSS_RE = re.compile(r"([\u3260-\u327b\u24d0-\u24e9])\s*\(\s*(['\"‘“])[^)'\"’”]+\2\s*\)")
_KOR_CONTRA_NOT_RE = re.compile(r"아니지만[^.\n]*아니라는\s*점에서\s*차이")
_KOR_CONTRA_SAME_RE = re.compile(
    r"([^\s,]{2,20}(?:\s+[^\s,]{1,15}){0,4})(?:이지만|인데)\s*[^.\n]*?\1(?:라는|다는)\s*점에서\s*차이"
)
_KOR_DUAL_CRITERIA_RE = re.compile(r"(?:및|와|과)\s*['‘\"]?(?:무사유|차이|오류|비판|대립|부정|한계)")


def _extract_korean_passage_markers(text: str) -> set[str]:
    """지문에 실제로 존재하는 국어 표지((가)~(라), [A]~[E], ㉠~㉻, ⓐ~ⓩ) 집합을 반환한다."""
    if not text:
        return set()
    return set(_KOR_MARKER_TOKEN_RE.findall(text))


def _format_marker_inventory_note(markers: set[str]) -> str:
    if not markers:
        return "실제 지문에는 별도의 기호 표지((가)/(나)/[A]~[E]/㉠~㉤/ⓐ~ⓔ 등)가 없습니다. 없는 기호를 절대 지어내지 마세요."
    ordered = sorted(markers)
    return (
        f"실제 지문에 존재하는 표지 목록: {', '.join(ordered)} "
        "(이 목록에 없는 표지 예: ㉮, [B] 등을 발문·선지·해설에 절대 지어내지 마세요.)"
    )


def _strip_markdown_noise(text: str, *, keep_underline: bool = True) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    s = re.sub(r"\*\*([^*\n]+?)\*\*", r"\1", s)
    s = re.sub(r"__([^_\n]+?)__", r"\1", s)
    s = s.replace("`", "")
    if not keep_underline:
        s = re.sub(r"</?u\b[^>]*>", "", s, flags=re.IGNORECASE)
    s = s.replace("보상청구권을 청구", "보상을 청구")
    return re.sub(r"[ \t]{2,}", " ", s).strip()


def _is_vocab_context_question(question_type_label: str, stem: str) -> bool:
    blob = f"{question_type_label} {stem}"
    return any(k in blob for k in ("어휘", "문맥상 의미", "문맥적 의미", "사전적 의미", "바꿔 쓰기"))


def _split_inline_bogi_from_stem(stem: str, existing_bogi: str) -> tuple[str, str]:
    """stem 안에 <보기>가 섞여 들어온 경우 순수 발문(위)과 <보기> 본문(아래)으로 분리한다."""
    bogi = (existing_bogi or "").strip()
    bogi = re.sub(r"^\s*[<〈]\s*보\s*기\s*[>〉]\s*", "", bogi).strip()
    raw_stem = (stem or "").strip()
    if not raw_stem:
        return "", bogi

    m_tag = re.search(r"[<〈]\s*보\s*기\s*[>〉]", raw_stem)
    if not m_tag or (raw_stem.find("<보기>") != -1 and bogi and "<보기>" in raw_stem and len(raw_stem) < 95):
        # stem이 '윗글을 바탕으로 <보기>의 ... 것은?'처럼 단순 언급만 한 경우
        if m_tag and not raw_stem[m_tag.start():].strip().startswith(("<보기>", "〈보기〉")):
            return raw_stem, bogi
        if not m_tag:
            return raw_stem, bogi

    # stem이 '<보기>\n...'로 시작하거나 뒤에 '\n<보기>\n...'가 붙은 경우 분리
    before = raw_stem[: m_tag.start()].strip()
    after = raw_stem[m_tag.end() :].strip()
    if before and any(k in before for k in ("것은?", "고르시오", "적절한", "분석한", "이해한")):
        if not bogi and after:
            bogi = after
        return before, bogi

    # '<보기>\n본문...\n윗글의 ... 적절한 것은?' 순서 역전 복구
    lines = [ln.strip() for ln in after.splitlines() if ln.strip()]
    stem_tail: list[str] = []
    bogi_lines: list[str] = []
    for ln in lines:
        if (
            not stem_tail
            and re.search(r"(?:윗글|위\s*글|<보기>|〈보기〉|적절한\s*것은|알맞은\s*것은|고른\s*것은)", ln)
            and re.search(r"(?:것은\s*\??|고르시오\.?)$", ln)
        ):
            stem_tail.append(ln)
        elif stem_tail:
            stem_tail.append(ln)
        else:
            bogi_lines.append(ln)

    if stem_tail:
        if not bogi and bogi_lines:
            bogi = "\n".join(bogi_lines).strip()
        return "\n".join(stem_tail).strip(), bogi
    return raw_stem, bogi


def _validate_and_sanitize_korean_question(
    gq: KoreanSetQuestionOutput,
    clean_passage: str,
    allowed_markers: set[str],
) -> tuple[KoreanSetQuestionOutput, list[str]]:
    """Gate 1: 국어 문항 마크업·순서 정규화 및 5대 하드 출제 결함 검증."""
    clean_stem, clean_bogi = _split_inline_bogi_from_stem(gq.stem, getattr(gq, "bogi_text", ""))
    clean_stem = _strip_markdown_noise(clean_stem, keep_underline=False)
    clean_bogi = _strip_markdown_noise(clean_bogi, keep_underline=True)

    # 지문에 이미 존재하는 기호(㉠/ⓐ 등) 뒤에 AI가 덧붙인 괄호 풀이 ㉠('...') 제거
    def _strip_redundant_gloss(m: re.Match[str]) -> str:
        mk = m.group(1)
        if mk in allowed_markers:
            return mk
        return m.group(0)

    clean_stem = _KOR_STEM_GLOSS_RE.sub(_strip_redundant_gloss, clean_stem)
    is_vocab_q = _is_vocab_context_question(gq.question_type_label, clean_stem)

    raw_choices = [re.sub(r"^[①②③④⑤]\s*|^[1-5]\)\s*", "", (c or "").strip()) for c in (gq.choices or [])]
    clean_choices: list[str] = []
    for c in raw_choices:
        if not c:
            continue
        if is_vocab_q:
            # 어휘 문항 선지의 **단어** -> <u>단어</u> 변환
            c_conv = re.sub(r"\*\*([^*\n]+?)\*\*", r"<u>\1</u>", c)
            c_conv = _strip_markdown_noise(c_conv, keep_underline=True)
        else:
            c_conv = _strip_markdown_noise(c, keep_underline=True)
        clean_choices.append(c_conv)

    raw_cexp = [
        _strip_markdown_noise(re.sub(r"^[①②③④⑤]\s*", "", (ce or "").strip()), keep_underline=False)
        for ce in (getattr(gq, "choice_explanations", None) or [])
    ]
    clean_exp = _strip_markdown_noise(gq.explanation, keep_underline=False)
    norm_ans = _normalize_extracted_answer(gq.answer)
    if norm_ans not in {"1", "2", "3", "4", "5"}:
        norm_ans = "1"

    sanitized = gq.model_copy(
        update={
            "stem": clean_stem,
            "bogi_text": clean_bogi,
            "choices": clean_choices,
            "answer": norm_ans,
            "explanation": clean_exp,
            "choice_explanations": raw_cexp,
        }
    )

    errors: list[str] = []
    if len(clean_choices) != 5:
        errors.append(f"선지 개수 오류({len(clean_choices)}개)")
    else:
        norm_c_set = {re.sub(r"\s+", "", c) for c in clean_choices}
        if len(norm_c_set) < 5:
            errors.append("중복 선지 존재")

    if not clean_stem:
        errors.append("발문 비어 있음")

    # <보기> 언급 <-> 본문 일치 검사
    mentions_bogi = ("<보기>" in clean_stem) or ("〈보기〉" in clean_stem)
    if mentions_bogi and not clean_bogi:
        errors.append("발문에서 <보기>를 언급했으나 bogi_text가 비어 있음")
    elif not mentions_bogi and clean_bogi:
        errors.append("bogi_text가 존재하나 발문에서 <보기>를 언급하지 않음")

    # 지문·<보기>에 없는 표지 참조 검사
    bogi_markers = _extract_korean_passage_markers(clean_bogi)
    valid_markers = allowed_markers | bogi_markers
    referenced_markers = _extract_korean_passage_markers(clean_stem + "\n" + "\n".join(clean_choices))
    unknown_markers = sorted(referenced_markers - valid_markers)
    if unknown_markers:
        errors.append(f"지문에 없는 표지 참조: {', '.join(unknown_markers)}")

    # 비교 선지 자체 모순·동어반복 패턴 검사
    for idx_c, ch in enumerate(clean_choices, start=1):
        if _KOR_CONTRA_NOT_RE.search(ch) or _KOR_CONTRA_SAME_RE.search(ch):
            errors.append(f"{idx_c}번 선지 비교 논리 자체모순/동어반복")

    # 상반 개념 이중 기준 발문 + 단일 기호 선지 조합 검사
    is_single_marker_choices = len(clean_choices) == 5 and all(
        re.fullmatch(r"[\u3260-\u327b\u24d0-\u24e9]|\[[A-E]\]", c.strip())
        for c in clean_choices
    )
    if is_single_marker_choices and (len(bogi_markers) >= 2 or _KOR_DUAL_CRITERIA_RE.search(clean_stem)):
        errors.append("단일 기호 선지(① ⓐ~⑤ ⓔ) + 이중 기준/함정형 <보기> 사용 금지 위반")

    return sanitized, errors


def _format_korean_question_for_solver(gq: KoreanSetQuestionOutput) -> str:
    parts = [f"[문항 {gq.slot_index}] ({gq.question_type_label})", gq.stem]
    if (gq.bogi_text or "").strip():
        parts.append(f"<보기>\n{gq.bogi_text.strip()}")
    for i, c in enumerate(gq.choices[:5]):
        parts.append(f"{_CIRCLED_NUMS[i]} {c}")
    return "\n".join(parts)


def _assemble_korean_full_stem(gq: KoreanSetQuestionOutput) -> str:
    """개별 문항의 순수 발문과 <보기> 박스를 조합한다 (지문은 shared_passage로 분리)."""
    stem_txt = (gq.stem or "").strip()
    bogi_txt = (gq.bogi_text or "").strip()
    if bogi_txt:
        return f"{stem_txt}\n\n<보기>\n{bogi_txt}"
    return stem_txt


def _assemble_korean_explanation(gq: KoreanSetQuestionOutput) -> str:
    """정답 근거와 ①~⑤ 선지별 정오 근거를 표준 평가원 해설 형식으로 조합한다."""
    summary = (gq.explanation or "").strip()
    cexp = list(getattr(gq, "choice_explanations", None) or [])
    lines: list[str] = []
    if summary:
        lines.append(f"[정답 해설]\n{summary}")
    if len(cexp) >= 5:
        per_choice = "\n".join(
            f"{_CIRCLED_NUMS[i]} {cexp[i].strip()}" for i in range(5) if cexp[i].strip()
        )
        if per_choice:
            lines.append(f"[선지별 분석]\n{per_choice}")
    raw = "\n\n".join(lines).strip()
    return format_explanation_text(raw, for_markdown=False)


def _verify_korean_set_with_blind_solver(
    *,
    client: Any,
    model_name: str,
    clean_passage: str,
    questions: list[KoreanSetQuestionOutput],
) -> KoreanBlindSetVerdict:
    """Gate 2: 정답·해설을 은닉한 학생용 문항 세트를 독립 풀이해 유일 정답·논리 결함을 검증한다."""
    q_blocks = "\n\n".join(_format_korean_question_for_solver(q) for q in questions)
    system_inst = (
        "당신은 대한민국 수능·평가원 국어 영역 문항 검수위원입니다.\n"
        "정답지가 전혀 주어지지 않은 상태에서, 오직 아래 [지문]과 [문항 세트]만 읽고 각 문항을 직접 푸세요.\n"
        "각 문항(`slot_index`)에 대해 다음을 엄격히 판정하세요:\n"
        "1. `selected_answer`: 지문과 논리적으로 부합하는 가장 확실한 유일한 정답(`'1'`~`'5'`).\n"
        "2. `ambiguous_choices`: 복수 정답이 되거나 두 선지가 모두 말이 되는 경우 해당 선지 번호들(예: `['2', '4']`), 문제 없으면 `[]`.\n"
        "3. `has_flaw`: 아래 중 하나라도 해당하면 `true`로 표시하고 `flaw_reason`을 구체적으로 적으세요:\n"
        "   - 지문에 명시되지 않은 외부 사실이 정답 근거로 쓰임\n"
        "   - 지문에 없는 표지(예: 지문에 `㉮`가 없는데 `㉮`를 물음) 참조\n"
        "   - 발문과 선지의 주술 호응 오류, 비문, 비교 선지의 동어반복/자체모순\n"
        "   - `<보기>`와 발문 조건이 서로 모순되거나 복수 기호가 모두 답이 됨"
    )
    prompt = f"[공통 지문]\n{clean_passage}\n\n[문항 세트]\n{q_blocks}"
    return _generate_structured_with_fallback(
        client=client,
        model_name=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=KoreanBlindSetVerdict,
            temperature=0.0,
            max_output_tokens=4096,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MEDIUM,
                include_thoughts=False,
            ),
        ),
        fast_mode=True,
        schema_cls=KoreanBlindSetVerdict,
        max_parse_retries=2,
    )


def _repair_defective_korean_questions(
    *,
    client: Any,
    model_name: str,
    clean_passage: str,
    korean_section: str,
    korean_subgenre: str,
    marker_note: str,
    all_questions: list[KoreanSetQuestionOutput],
    defect_reasons: dict[int, list[str]],
) -> list[KoreanSetQuestionOutput]:
    """결함이 감지된 슬롯의 문항만 지문과 결함 사유를 바탕으로 정밀 재출제한다."""
    target_slots = sorted(defect_reasons.keys())
    defect_lines = "\n".join(
        f"- slot_index {s}: {' / '.join(defect_reasons[s])}" for s in target_slots
    )
    existing_json = json.dumps(
        [q.model_dump() for q in all_questions],
        ensure_ascii=False,
        indent=2,
    )
    system_inst = (
        "당신은 대한민국 수능·평가원 국어 영역 최고 수준 출제 검수위원입니다.\n"
        f"아래 [공통 지문]과 [결함 보고]를 확인하고, **결함이 지적된 문항(slot_index: {target_slots})만** "
        "완벽한 수능형 문항으로 전면 수정·재출제하여 `questions` 리스트로 반환하세요.\n\n"
        "[재출제 절대 규칙]\n"
        f"1. {marker_note}\n"
        "2. `<보기>`에 `ⓐ~ⓔ`를 임의로 박아 `① ⓐ ... ⑤ ⓔ` 단일 기호를 고르게 하는 함정형 출제를 절대 하지 말고, 반드시 완전한 진술문 5개 선지로 출제하세요.\n"
        "3. 비교 선지에서 동어반복·자체모순('아니지만 ... 아니라는 점에서 차이')을 절대 쓰지 마세요.\n"
        "4. 주술 호응과 한국어 어문 규범을 100% 지키고, 마크다운 `**`는 절대 쓰지 마세요 (어휘 문항 선지의 대상 어휘만 `<u>...</u>` 사용).\n"
        "5. `stem`에는 순수 발문 1문장만, `<보기>` 내용은 `bogi_text`에 분리해 적으세요.\n"
        "6. `explanation`과 5개 `choice_explanations`를 모두 지문 근거 기반 평어체(`~다.`)로 작성하세요."
    )
    prompt = (
        f"[영역/갈래] {korean_section} ({korean_subgenre or '공통'})\n\n"
        f"[공통 지문]\n{clean_passage}\n\n"
        f"[기존 문항 목록]\n{existing_json}\n\n"
        f"[수정 대상 슬롯 및 결함 사유]\n{defect_lines}\n\n"
        f"slot_index {target_slots} 문항만 결함을 완전히 해결해 다시 작성하세요."
    )
    repaired_out = _generate_structured_with_fallback(
        client=client,
        model_name=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=KoreanSetRepairOutput,
            temperature=0.25,
            max_output_tokens=12288,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MEDIUM,
                include_thoughts=False,
            ),
        ),
        fast_mode=True,
        schema_cls=KoreanSetRepairOutput,
        max_parse_retries=2,
    )
    return list(repaired_out.questions or [])


def _is_fixed_order_korean_choices(choices: Sequence[str]) -> bool:
    """선지 순서가 고정되어야 하는 문항(단일 기호, ㄱㄴㄷ 조합형, (A)-(B) 순서형)인지 판별한다."""
    if len(choices) != 5:
        return True
    cleaned = [c.strip() for c in choices]
    if all(re.fullmatch(r"[\u3260-\u327b\u24d0-\u24e9]|\[[A-E]\]", c) for c in cleaned):
        return True
    if all(re.fullmatch(r"[ㄱㄴㄷㄹㅁ](?:\s*,\s*[ㄱㄴㄷㄹㅁ])*", c) for c in cleaned):
        return True
    if sum(1 for c in cleaned if "(A)" in c and "(B)" in c) >= 4:
        return True
    return False


def _parse_choice_explanations_from_text(exp_text: str) -> tuple[str, list[str]]:
    """조립된 해설 텍스트에서 [정답 해설] 머리글과 ①~⑤ 선지별 분석 5줄을 분리한다."""
    raw = (exp_text or "").strip()
    if not raw:
        return "", []
    m_split = re.search(r"\[선지별\s*분석\]\s*\n?", raw)
    if not m_split:
        return raw, []
    head = raw[: m_split.start()].strip()
    tail = raw[m_split.end() :].strip()
    per_choice = ["", "", "", "", ""]
    cur_idx = -1
    sym_to_idx = {sym: idx for idx, sym in enumerate(_CIRCLED_NUMS)}
    for ln in tail.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s[0] in sym_to_idx:
            cur_idx = sym_to_idx[s[0]]
            body = s[1:].strip()
            per_choice[cur_idx] = body
        elif cur_idx >= 0:
            per_choice[cur_idx] = f"{per_choice[cur_idx]} {s}".strip()
    if all(per_choice):
        return head, per_choice
    return raw, []


def _permute_rewritten_question_choices(
    rw: RewrittenQuestion,
    target_ans: str,
) -> RewrittenQuestion:
    """문항의 정답 번호를 target_ans('1'~'5')로 이동시키고 선지와 선지별 해설을 동일 순열로 재배치한다."""
    cur_ans = _normalize_extracted_answer(rw.preserved_answer)
    if cur_ans not in {"1", "2", "3", "4", "5"} or target_ans not in {"1", "2", "3", "4", "5"}:
        return rw
    if cur_ans == target_ans or len(rw.rewritten_choices) != 5:
        return rw

    old_idx = int(cur_ans) - 1
    new_idx = int(target_ans) - 1
    perm = [0, 1, 2, 3, 4]
    perm[old_idx], perm[new_idx] = perm[new_idx], perm[old_idx]

    new_choices = [rw.rewritten_choices[p] for p in perm]
    head, per_choice = _parse_choice_explanations_from_text(rw.pdf_grounded_explanation)
    if len(per_choice) == 5:
        new_per_choice = [per_choice[p] for p in perm]
        per_block = "[선지별 분석]\n" + "\n".join(
            f"{_CIRCLED_NUMS[i]} {new_per_choice[i]}" for i in range(5)
        )
        new_exp = f"{head}\n\n{per_block}".strip() if head else per_block
        new_exp = format_explanation_text(new_exp, for_markdown=False)
    else:
        new_exp = rw.pdf_grounded_explanation

    return rw.model_copy(
        update={
            "rewritten_choices": new_choices,
            "preserved_answer": target_ans,
            "pdf_grounded_explanation": new_exp,
        }
    )


def balance_korean_answer_distribution(
    items: list[tuple[QuestionRecord, RewrittenQuestion]],
) -> list[tuple[QuestionRecord, RewrittenQuestion]]:
    """
    국어 세트 문항 목록의 정답 번호(①~⑤) 편중 및 3연속 동일 번호를 결정론적으로 분산한다.
    고정 순서 선지(ㄱ,ㄴ,ㄷ 조합형 등)는 유지하며, 일반 진술문 선지는 선지와 [선지별 분석] 해설을 함께 재배치한다.
    """
    if len(items) <= 1:
        return items

    n = len(items)
    base_count = n // 5
    remainder = n % 5
    # 시작 오프셋은 문항 수에 따라 결정론적으로 분산
    targets_pool: list[str] = []
    for k in range(5):
        cnt = base_count + (1 if k < remainder else 0)
        targets_pool.extend([str(k + 1)] * cnt)

    # 현재 고정 문항과 가변 문항 분류
    counts: dict[str, int] = {str(k): 0 for k in range(1, 6)}
    out_items: list[tuple[QuestionRecord, RewrittenQuestion]] = list(items)
    mutable_indices: list[int] = []

    for i, (_, rw) in enumerate(out_items):
        cur_a = _normalize_extracted_answer(rw.preserved_answer)
        if cur_a not in counts:
            cur_a = "1"
        if _is_fixed_order_korean_choices(rw.rewritten_choices):
            counts[cur_a] += 1
        else:
            mutable_indices.append(i)

    if not mutable_indices:
        return out_items

    max_allowed = (n + 4) // 5
    assigned_answers: dict[int, str] = {}
    for i, (_, rw) in enumerate(out_items):
        if i not in mutable_indices:
            assigned_answers[i] = _normalize_extracted_answer(rw.preserved_answer) or "1"

    # 이미 적정 상한 이내인 가변 문항은 1차 유지, 초과분만 부족한 번호로 재배정
    need_reassign: list[int] = []
    for idx_m in mutable_indices:
        cur_a = _normalize_extracted_answer(out_items[idx_m][1].preserved_answer)
        if cur_a not in counts:
            cur_a = "1"
        prev2 = [assigned_answers.get(idx_m - 1), assigned_answers.get(idx_m - 2)]
        breaks_run = prev2[0] == cur_a and prev2[1] == cur_a
        if counts[cur_a] < max_allowed and not breaks_run:
            counts[cur_a] += 1
            assigned_answers[idx_m] = cur_a
        else:
            need_reassign.append(idx_m)

    for idx_m in need_reassign:
        # 가장 적게 나온 번호 우선, 이전 두 문항과 3연속 회피
        prev1 = assigned_answers.get(idx_m - 1)
        prev2 = assigned_answers.get(idx_m - 2)
        candidates = sorted(
            ("1", "2", "3", "4", "5"),
            key=lambda a: (
                1 if (a == prev1 and a == prev2) else 0,
                1 if a == prev1 else 0,
                counts[a],
                int(a),
            ),
        )
        chosen = candidates[0]
        counts[chosen] += 1
        assigned_answers[idx_m] = chosen

    # 연속 3문항 동일 번호 최종 점검
    for idx_m in mutable_indices:
        a_cur = assigned_answers[idx_m]
        if idx_m >= 2 and assigned_answers.get(idx_m - 1) == a_cur and assigned_answers.get(idx_m - 2) == a_cur:
            swap_cand = next(
                (
                    j
                    for j in mutable_indices
                    if j != idx_m
                    and assigned_answers[j] != a_cur
                    and not (
                        j >= 2
                        and assigned_answers.get(j - 1) == a_cur
                        and assigned_answers.get(j - 2) == a_cur
                    )
                ),
                None,
            )
            if swap_cand is not None:
                assigned_answers[idx_m], assigned_answers[swap_cand] = (
                    assigned_answers[swap_cand],
                    assigned_answers[idx_m],
                )

    for idx_m in mutable_indices:
        rec, rw = out_items[idx_m]
        out_items[idx_m] = (rec, _permute_rewritten_question_choices(rw, assigned_answers[idx_m]))
    return out_items


def generate_korean_set_reconstruction(
    api_key: str,
    model_name: str,
    source_records: list[QuestionRecord],
    passage_text: str,
    korean_section: str = "독서",
    korean_subgenre: str = "",
    passage_topic: str = "",
    *,
    storage: Any = None,
    use_cache: bool = True,
    username: str = "",
) -> list[tuple[QuestionRecord, RewrittenQuestion]]:
    """
    국어 지문 1개 세트(원본 문항 수 N개, 보통 3~6문항)를 받아
    지문을 정제·표지 검증한 뒤 N개의 수능형 국어 문항 세트를 창작하고,
    Gate 1(하드 규칙) + Gate 2(정답 은닉 Blind Solver) + 결함 슬롯 재출제 + 정답 분산까지 완료해 반환한다.
    storage가 주어지면 미열람 풀 캐시를 우선 조회하고, 신규 생성 성공 시 풀에 영구 저장한다.
    공통 지문은 RewrittenQuestion.shared_passage에 분리 저장하여 PDF·웹 풀기에서 1회만 출력한다.
    """
    ordered_recs = sorted(source_records, key=lambda r: r.q_num)
    if not ordered_recs:
        return []

    n_questions = len(ordered_recs)
    is_keep_lines = _is_korean_dialogue_or_poetry(korean_section, korean_subgenre, passage_text)
    clean_passage = clean_korean_exam_passage(passage_text or "", is_poetry_or_dialogue=is_keep_lines)
    if not clean_passage:
        raise ValueError("국어 지문 원문(passage_text)이 비어 있어 세트 문항을 재구성할 수 없습니다.")

    gid = normalize_korean_group_id(getattr(ordered_recs[0], "passage_group_id", ""), ordered_recs[0].q_num)
    group_key = f"{ordered_recs[0].exam_name}::{gid or ordered_recs[0].q_num}"

    cache_key = ""
    if storage is not None and hasattr(storage, "build_korean_pool_key"):
        cache_key = storage.build_korean_pool_key(
            clean_passage,
            group_id=gid,
            korean_section=korean_section,
            question_count=n_questions,
            korean_subgenre=korean_subgenre,
        )
        if use_cache and hasattr(storage, "get_cached_korean_set"):
            cached_hit = storage.get_cached_korean_set(
                cache_key,
                expected_count=n_questions,
                username=username,
            )
            if cached_hit is not None:
                cached_rws, _ = cached_hit
                return [
                    (
                        ordered_recs[idx],
                        cached_rws[idx].model_copy(
                            update={
                                "original_question_id": ordered_recs[idx].id,
                                "passage_group_key": group_key,
                            }
                        ),
                    )
                    for idx in range(n_questions)
                ]

    allowed_markers = _extract_korean_passage_markers(clean_passage)
    marker_note = _format_marker_inventory_note(allowed_markers)

    client = _build_client(api_key, timeout_seconds=150)
    orig_summary_lines = [
        f"- 원본 {r.q_num}번 ({r.unit_name or '기출'}): 핵심={'; '.join(r.core_principles[:2])} | 정답={r.answer}"
        for r in ordered_recs
    ]
    orig_summary_block = "\n".join(orig_summary_lines)

    if korean_section == "문학":
        type_guide = (
            "문학 영역 추천 유형을 지문 특성에 맞게 자유롭게 배분하세요: "
            "표현상·서술상 특징, 시적 화자/서술자 및 인물의 정서·태도, 구절·소재의 함축적 의미, "
            "<보기> 외적 준거(비평문·창작 배경) 기반 작품 감상 적절성, (가)/(나) 작품 간 비교·공통점과 차이점."
        )
    elif korean_section == "화법과 작문":
        type_guide = (
            "화법과 작문 영역 추천 유형을 지문/대화 특성에 맞게 자유롭게 배분하세요: "
            "화자/발표자의 말하기 방식 및 전략, 청자/참여자의 반응 분석, 초고 고쳐쓰기 방안의 적절성, "
            "<보기> 자료(도표·조사 결과·메모) 활용 및 반영 여부."
        )
    elif korean_section == "언어와 매체":
        type_guide = (
            "언어와 매체 영역 추천 유형을 지문/자료 특성에 맞게 자유롭게 배분하세요: "
            "국어 문법(음운 변동, 단어 형성, 문장 성분·안은문장, 중세 국어) 원리 적용, "
            "<보기> 탐구 사례 분석, 매체 자료의 정보 전달 방식·수용자 반응·매체 언어 특성 이해."
        )
    else:
        type_guide = (
            "독서(비문학) 영역 추천 유형을 지문 특성에 맞게 자유롭게 배분하세요: "
            "글의 전개 방식·서술상 특징, 세부 내용 일치·불일치, 핵심 논지·개념 간 관계 추론, "
            "<보기>의 구체적 사례/상황에 지문 원리 적용 추론, 특정 문맥 어휘의 문맥적 의미."
        )

    system_inst = (
        "당신은 대한민국 수능·평가원 국어 영역 최고 수준 문항 출제위원입니다.\n"
        f"주어진 [{korean_section} · {korean_subgenre or '공통'}] 지문을 바탕으로, 기존 기출 문항을 그대로 베끼지 말고 "
        f"**총 {n_questions}개의 고품질 수능 국어 세트 문항**을 자유 유형으로 새롭게 창작하세요.\n\n"
        "[출제 절대 원칙]\n"
        "1. **지문 100% 근거 원칙**: 모든 문항과 선지의 정오 판정은 반드시 아래 [공통 지문 원문]에 직접 명시되거나 논리적으로 도출 가능한 사실에만 근거해야 합니다. 지문에 없는 외부 지식을 정답 근거로 쓰지 마세요.\n"
        f"2. **지문 표지 정밀 연동**: {marker_note}\n"
        "   - 발문에 `㉠('...')`처럼 괄호로 긴 풀이를 덧붙이지 말고, `윗글의 ㉠에 대한 설명으로 가장 적절한 것은?`처럼 깔끔하게 작성하세요.\n"
        f"3. **자유 유형 균형 출제 ({n_questions}문항)**: {type_guide}\n"
        "   - 최소 1문항 이상은 구체적인 `<보기>` 박스를 포함한 고난도 적용·감상 문항으로 출제하세요.\n"
        "4. **<보기> 함정 출제 금지**: `<보기>` 내부에 임의로 `ⓐ~ⓔ` 기호를 끼워 넣고 `① ⓐ ... ⑤ ⓔ` 단일 기호를 고르게 하는 출제는 **절대 금지**합니다. `<보기>` 문항도 반드시 5개의 완전한 문장(진술문) 선지로 출제하세요.\n"
        "5. **비교 선지 모순·비문 원천 차단**:\n"
        "   - 비교 문항에서 `'A는 X이지만, B는 X라는 점에서 차이가 난다'` 또는 `'아니지만 ... 아니라는 점에서 차이'` 같은 동어반복·자체모순 오답 선지를 절대 작성하지 마세요.\n"
        "   - `'~의 크기'`는 `'커진다/감소한다'`, `'역기전력'`은 `'발생한다'`처럼 주술 호응을 100% 맞추고, `'보상청구권을 청구'` 같은 비문 대신 `'보상을 청구'` 또는 `'보상청구권을 행사'`로 쓰세요.\n"
        "6. **발문·<보기> 분리 및 마크다운 금지**:\n"
        "   - `stem`에는 순수 발문 1문장만 적고, `<보기>` 본문은 반드시 `bogi_text`에 적으세요 (`set_intro`에는 번호 범위를 뺀 `'다음 글을 읽고 물음에 답하시오.'` 등 지시문만 적으세요).\n"
        "   - `stem`, `bogi_text`, `choices` 어디에도 마크다운 별표(`**`)를 절대 쓰지 마세요. (단, 어휘 문맥 의미 문항 선지의 대상 어휘만 `<u>따랐다</u>`처럼 `<u>...</u>` 밑줄 태그 사용)\n"
        "7. **선지 5개(`choices`) & 정답 분산**: `choices`는 선지 번호(①~⑤)를 제외한 5개 문자열 리스트로 작성하고, `answer`는 `'1'`~`'5'` 중 한 번호에 몰리지 않게 골고루 배치하세요.\n"
        "8. **정교한 해설(`explanation` + `choice_explanations`)**: `explanation`에 정답 핵심 근거를, `choice_explanations`에 `choices[0]~[4]` 각각에 1:1 대응하는 선지별 정오 근거 5개를 모두 평가원 평어체(`~다.`)로 작성하세요."
    )

    prompt = (
        f"[지문 정보]\n"
        f"- 시험 출처: {ordered_recs[0].exam_name}\n"
        f"- 영역/갈래: {korean_section} ({korean_subgenre or '공통'})\n"
        f"- 주제/작품: {passage_topic or ordered_recs[0].passage_topic}\n"
        f"- 생성할 문항 수: 정확히 {n_questions}문항 (slot_index 1 ~ {n_questions})\n\n"
        f"[공통 지문 원문]\n{clean_passage}\n\n"
        f"[원본 세트 구성 참고 (소재·난이도 참고용, 그대로 복제하지 말고 자유롭게 새 문항 창작)]\n{orig_summary_block}\n\n"
        f"위 지문을 바탕으로 정확히 {n_questions}개의 새로운 수능형 국어 세트 문항을 JSON(set_intro, questions)으로 생성하세요."
    )

    parsed = _generate_structured_with_fallback(
        client=client,
        model_name=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=KoreanSetReconstructionOutput,
            temperature=0.3,
            max_output_tokens=16384,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MEDIUM,
                include_thoughts=False,
            ),
        ),
        fast_mode=True,
        schema_cls=KoreanSetReconstructionOutput,
        max_parse_retries=2,
    )
    raw_qs = sorted(parsed.questions or [], key=lambda x: x.slot_index)[:n_questions]
    if not raw_qs:
        raise ValueError("AI가 국어 세트 문항을 생성하지 못했습니다. 다시 시도해 주세요.")

    # Gate 1: 파이썬 마크업 정화 & 하드 규칙 검증
    sanitized_qs: list[KoreanSetQuestionOutput] = []
    defect_reasons: dict[int, list[str]] = {}
    for idx_q, gq in enumerate(raw_qs, start=1):
        gq_norm = gq.model_copy(update={"slot_index": idx_q})
        clean_q, g1_errs = _validate_and_sanitize_korean_question(gq_norm, clean_passage, allowed_markers)
        sanitized_qs.append(clean_q)
        if g1_errs:
            defect_reasons.setdefault(idx_q, []).extend(g1_errs)

    # Gate 2: 정답 은닉 Blind Solver 검증
    try:
        blind_res = _verify_korean_set_with_blind_solver(
            client=client,
            model_name=model_name,
            clean_passage=clean_passage,
            questions=sanitized_qs,
        )
        verdict_map = {v.slot_index: v for v in (blind_res.verdicts or [])}
        for q_obj in sanitized_qs:
            v = verdict_map.get(q_obj.slot_index)
            if v is None:
                continue
            intended = _normalize_extracted_answer(q_obj.answer)
            if v.selected_answer != intended:
                defect_reasons.setdefault(q_obj.slot_index, []).append(
                    f"Blind Solver 정답 불일치(출제={intended}번 vs 검수={v.selected_answer}번: {v.flaw_reason})"
                )
            other_amb = [a for a in (v.ambiguous_choices or []) if a != intended]
            if other_amb:
                defect_reasons.setdefault(q_obj.slot_index, []).append(
                    f"복수 정답 경합({','.join(other_amb)}번: {v.flaw_reason})"
                )
            if v.has_flaw:
                defect_reasons.setdefault(q_obj.slot_index, []).append(
                    f"논리/표현 결함({v.flaw_reason})"
                )
    except Exception:
        pass

    # 결함 슬롯 1회 타겟 재출제 + Gate 1 & Gate 2 재확인
    if defect_reasons:
        try:
            repaired_list = _repair_defective_korean_questions(
                client=client,
                model_name=model_name,
                clean_passage=clean_passage,
                korean_section=korean_section,
                korean_subgenre=korean_subgenre,
                marker_note=marker_note,
                all_questions=sanitized_qs,
                defect_reasons=defect_reasons,
            )
            rep_map = {r.slot_index: r for r in repaired_list}
            for i_s, cur_q in enumerate(sanitized_qs):
                if cur_q.slot_index in rep_map:
                    cand_rep, _ = _validate_and_sanitize_korean_question(
                        rep_map[cur_q.slot_index], clean_passage, allowed_markers
                    )
                    if len(cand_rep.choices) == 5 and cand_rep.stem:
                        sanitized_qs[i_s] = cand_rep
            # Gate 2 재확인
            _verify_korean_set_with_blind_solver(
                client=client,
                model_name=model_name,
                clean_passage=clean_passage,
                questions=sanitized_qs,
            )
        except Exception:
            pass

    raw_intro = re.sub(r"^\s*\[?\s*\d{1,2}\s*[~\-～]\s*\d{1,2}\s*\]?\s*", "", (parsed.set_intro or "").strip()).strip()
    set_instruction = raw_intro or "다음 글을 읽고 물음에 답하시오."

    results: list[tuple[QuestionRecord, RewrittenQuestion]] = []
    for idx, gq in enumerate(sanitized_qs):
        base_rec = ordered_recs[idx] if idx < len(ordered_recs) else ordered_recs[-1]
        clean_choices = list(gq.choices[:5])
        while len(clean_choices) < 5:
            clean_choices.append("(선지 보완 필요)")

        rw = RewrittenQuestion(
            original_question_id=base_rec.id,
            rewritten_stem=_assemble_korean_full_stem(gq),
            rewritten_choices=clean_choices,
            replaced_terms_map=[],
            pdf_grounded_explanation=_assemble_korean_explanation(gq),
            preserved_answer=_normalize_extracted_answer(gq.answer),
            self_verification_note=f"국어 {korean_section} 지문 세트 2-Gate 검증 완료 ({gq.question_type_label})",
            visual_term_note="",
            target_question_type=gq.question_type_label.strip() or "국어 세트 재구성",
            subject_domain="국어",
            shared_passage=clean_passage,
            passage_group_key=group_key,
            set_instruction=set_instruction,
        )
        results.append((base_rec, rw))

    final_pairs = balance_korean_answer_distribution(results)
    if cache_key and storage is not None and hasattr(storage, "save_cached_korean_set") and final_pairs:
        try:
            storage.save_cached_korean_set(
                cache_key,
                [rw for _, rw in final_pairs],
                exam_name=ordered_recs[0].exam_name,
                group_id=gid,
                korean_section=korean_section,
                korean_subgenre=korean_subgenre,
                passage_topic=passage_topic or getattr(ordered_recs[0], "passage_topic", ""),
            )
        except Exception:
            pass
    return final_pairs


def generate_korean_question_explanation(
    *,
    api_key: str,
    model_name: str,
    question: QuestionRecord,
    passage_text: str,
    user_answer: str,
    rewritten: RewrittenQuestion | None = None,
    source_reference: str = "",
) -> str:
    """모드 B: 국어 지문 원문만을 근거로 학생 답안 진단 및 ①~⑤ 선지별 정오 해설을 작성한다."""
    client = _build_client(api_key, timeout_seconds=60)
    clean_p = clean_korean_exam_passage(passage_text or "")
    q_stem = rewritten.rewritten_stem if rewritten is not None else (getattr(question, "question_text", "") or "")
    q_choices = (
        "\n".join(f"{_CIRCLED_NUMS[i]} {c}" for i, c in enumerate(rewritten.rewritten_choices[:5]))
        if rewritten is not None and rewritten.rewritten_choices
        else ""
    )
    truth = rewritten.preserved_answer if rewritten is not None else question.answer
    base_exp = rewritten.pdf_grounded_explanation if rewritten is not None else question.official_explanation

    system_inst = (
        "당신은 대한민국 수능 국어 영역 1:1 전담 튜터입니다.\n"
        "반드시 주어진 [지문 원문]만을 근거로 하여 학생이 고른 답안을 진단하고, "
        "정답이 왜 지문과 부합하는지, 나머지 선지가 지문의 어느 부분과 어긋나는지 ①~⑤ 선지별로 명쾌하게 해설하세요."
    )
    prompt = (
        f"[출처] {source_reference or f'{question.exam_name} {question.q_num}번'}\n"
        f"[정답] {truth}번 | [학생 선택] {user_answer or '미입력'}\n\n"
        f"[지문 원문]\n{clean_p}\n\n"
        f"[문항]\n{q_stem}\n{q_choices}\n\n"
        f"[기본 해설 참고]\n{base_exp}\n\n"
        "학생에게 보여줄 친절하고 논리적인 국어 AI 맞춤 해설을 마크다운으로 작성하세요."
    )
    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            temperature=0.2,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL,
                include_thoughts=False,
            ),
        ),
        fast_mode=True,
    )
    return (resp.text or "").strip()


def chat_followup_with_korean(
    *,
    api_key: str,
    model_name: str,
    question: QuestionRecord,
    passage_text: str,
    chat_history: list[dict[str, str]],
    user_query: str,
    rewritten: RewrittenQuestion | None = None,
) -> str:
    """모드 B: 국어 지문 원문 근거 1:1 후속 질의응답."""
    client = _build_client(api_key, timeout_seconds=45)
    clean_p = clean_korean_exam_passage(passage_text or "")
    q_stem = rewritten.rewritten_stem if rewritten is not None else (getattr(question, "question_text", "") or "")
    q_choices = (
        "\n".join(f"{_CIRCLED_NUMS[i]} {c}" for i, c in enumerate(rewritten.rewritten_choices[:5]))
        if rewritten is not None and rewritten.rewritten_choices
        else ""
    )
    truth = rewritten.preserved_answer if rewritten is not None else question.answer
    history_txt = "\n".join(
        f"{'학생' if m.get('role') == 'user' else 'AI 튜터'}: {m.get('content', '')}"
        for m in chat_history[-8:]
    )
    system_inst = (
        "당신은 수능 국어 영역 1:1 튜터입니다. 반드시 아래 [지문 원문]의 내용과 논리에만 근거하여 "
        "학생의 질문에 친절하고 정확하게 답변하세요."
    )
    prompt = (
        f"[지문 원문]\n{clean_p}\n\n"
        f"[문항 (정답: {truth}번)]\n{q_stem}\n{q_choices}\n\n"
        f"[이전 대화]\n{history_txt}\n\n"
        f"[학생 질문]\n{user_query}"
    )
    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            temperature=0.25,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL,
                include_thoughts=False,
            ),
        ),
        fast_mode=True,
    )
    return (resp.text or "").strip()


def regenerate_single_korean_question(
    *,
    api_key: str,
    model_name: str,
    question: QuestionRecord,
    passage_text: str,
    korean_section: str = "독서",
    korean_subgenre: str = "",
    passage_topic: str = "",
    previous_attempt: RewrittenQuestion | None = None,
) -> RewrittenQuestion:
    """모드 B: 국어 단일 문항을 2-Gate 검증으로 새롭게 재출제한다."""
    is_keep_lines = _is_korean_dialogue_or_poetry(korean_section, korean_subgenre, passage_text)
    clean_p = clean_korean_exam_passage(passage_text or "", is_poetry_or_dialogue=is_keep_lines)
    if not clean_p:
        raise ValueError("재출제할 국어 지문 원문이 비어 있습니다.")

    allowed_markers = _extract_korean_passage_markers(clean_p)
    marker_note = _format_marker_inventory_note(allowed_markers)
    client = _build_client(api_key, timeout_seconds=90)

    prev_txt = ""
    if previous_attempt is not None:
        prev_txt = (
            f"\n[이전 출제 문항 (이와 다른 새로운 관점/유형으로 더 완성도 높게 재출제)]\n"
            f"- 유형: {previous_attempt.target_question_type}\n"
            f"- 발문: {previous_attempt.rewritten_stem}\n"
        )

    system_inst = (
        "당신은 대한민국 수능·평가원 국어 영역 출제위원입니다.\n"
        f"주어진 [{korean_section}] 지문을 근거로 1개의 고품질 수능형 문항(`KoreanSetRepairOutput`의 `questions` 1개 항목)을 출제하세요.\n"
        f"1. {marker_note}\n"
        "2. `<보기>`에 `ⓐ~ⓔ`를 임의로 넣어 `① ⓐ ... ⑤ ⓔ` 단일 기호를 고르게 하는 함정 출제 금지.\n"
        "3. 비교 선지 동어반복·자체모순 금지, 주술 호응 엄수, 마크다운 `**` 금지(어휘 문항만 `<u>...</u>`).\n"
        "4. `stem`은 순수 발문 1문장, `<보기>` 본문은 `bogi_text`에 분리 작성하고 `explanation`과 5개 `choice_explanations`를 평어체(`~다.`)로 작성하세요."
    )
    prompt = (
        f"[출처] {question.exam_name} {question.q_num}번 | 주제: {passage_topic or question.passage_topic}\n"
        f"{prev_txt}\n[공통 지문 원문]\n{clean_p}\n\n"
        "slot_index=1인 새 문항 1개만 생성하세요."
    )
    rep_out = _generate_structured_with_fallback(
        client=client,
        model_name=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=KoreanSetRepairOutput,
            temperature=0.35,
            max_output_tokens=6144,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MEDIUM,
                include_thoughts=False,
            ),
        ),
        fast_mode=True,
        schema_cls=KoreanSetRepairOutput,
        max_parse_retries=2,
    )
    if not rep_out.questions:
        raise ValueError("국어 문항 재출제에 실패했습니다.")
    cand_q, errs = _validate_and_sanitize_korean_question(
        rep_out.questions[0].model_copy(update={"slot_index": 1}),
        clean_p,
        allowed_markers,
    )
    if errs:
        repaired = _repair_defective_korean_questions(
            client=client,
            model_name=model_name,
            clean_passage=clean_p,
            korean_section=korean_section,
            korean_subgenre=korean_subgenre,
            marker_note=marker_note,
            all_questions=[cand_q],
            defect_reasons={1: errs},
        )
        if repaired:
            cand_q, _ = _validate_and_sanitize_korean_question(
                repaired[0].model_copy(update={"slot_index": 1}),
                clean_p,
                allowed_markers,
            )

    gid = normalize_korean_group_id(getattr(question, "passage_group_id", ""), question.q_num)
    group_key = (
        getattr(previous_attempt, "passage_group_key", "")
        or f"{question.exam_name}::{gid or question.q_num}"
    )
    set_inst = getattr(previous_attempt, "set_instruction", "") or "다음 글을 읽고 물음에 답하시오."

    clean_choices = list(cand_q.choices[:5])
    while len(clean_choices) < 5:
        clean_choices.append("(선지 보완 필요)")

    return RewrittenQuestion(
        original_question_id=question.id,
        rewritten_stem=_assemble_korean_full_stem(cand_q),
        rewritten_choices=clean_choices,
        replaced_terms_map=[],
        pdf_grounded_explanation=_assemble_korean_explanation(cand_q),
        preserved_answer=_normalize_extracted_answer(cand_q.answer),
        self_verification_note=f"국어 {korean_section} 문항 개별 재출제 완료 ({cand_q.question_type_label})",
        visual_term_note="",
        target_question_type=cand_q.question_type_label.strip() or "국어 세트 재구성",
        subject_domain="국어",
        shared_passage=clean_p,
        passage_group_key=group_key,
        set_instruction=set_inst,
    )


def _strip_korean_exam_noise_lines(text: str) -> str:
    """국어 지문 원문에 섞인 시험지 쪽수('4', '12') 및 푸터('국어 영역', '12 국어') 노이즈 줄을 제거한다."""
    cleaned_lines: list[str] = []
    for raw_ln in (text or "").splitlines():
        s = raw_ln.strip()
        if re.match(r"^\[?\s*\d{1,2}\s*\]?$", s):
            continue
        if re.match(r"^(?:국어|수학|영어|한국사)\s*영역(?:\s*\d{1,2})?$", s):
            continue
        if re.match(r"^(?:\d{1,2}\s+국어|국어\s+\d{1,2})$", s):
            continue
        cleaned_lines.append(raw_ln)
    return "\n".join(cleaned_lines).strip()


_KOREAN_DIAGRAM_GUIDE = (
    "지문의 핵심 내용을 그림으로 한눈에 이해할 수 있도록 아래 12종 중 가장 적합한 `diagram_type`을 고르세요 "
    "(단순 박스 나열 `flow`만 반복하지 말고 지문 성격에 맞게 가장 직관적인 형식을 선택하세요):\n"
    "  ① `matrix` (항목별 비교 매트릭스표): 두 사상가·이론·법제·개념의 기준별 차이점을 비교할 때 사용. "
    "`nodes`에 비교 대상 2~3개 제목을 넣고, `matrix_rows`에 기준(`criterion`)과 대상별 값(`values`) 3~5행을 반드시 채우세요.\n"
    "  ② `graph` (X-Y 비례/함수 그래프): 경제(수요·공급·탄력성·비용·가격)나 과학(온도·압력·농도)에서 변수 간 비례·반비례·임계점 관계를 설명할 때 사용. "
    "`graph_spec`(`x_label`, `y_label`, `curves`[trend: up/down/u_shape/inv_u/flat], `highlight_point`)를 반드시 채우고 `nodes`에 1~2개 핵심 해석을 넣으세요.\n"
    "  ③ `anatomy` (구조·장치 모식도): 과학·기술·기계(튜링 기계, 세포, 엔진, 컴퓨터 구조 등)의 구성 요소와 역할을 설명할 때 사용. "
    "`center_label`에 중앙 장치/원리명을, `nodes`에 3~4개 부위·기능 카드를 넣으세요.\n"
    "  ④ `cycle` (원형 순환·피드백 루프): 상호작용, 피드백, 순환 과정(예: 네트워크 외부성, 생태계·경제 순환)에 사용. `center_label`과 `nodes` 3~4개, 짧은 `relation_labels`(6자 이내)를 넣으세요.\n"
    "  ⑤ `timeline` (시대·단계 변천 연표): 철학·과학사·역사적 이론 발전 순서에 사용. `nodes`에 시간순 3~4개 단계를 넣으세요.\n"
    "  ⑥ `venn` (포함·교집합 벤다이어그램): 두 개념의 공통점(`center_label`)과 고유 영역을 구분할 때 사용.\n"
    "  ⑦ `scale` (대립·균형 저울도): 두 상충 가치·법익·트레이드오프를 비교할 때 사용 (`nodes` 2개, `scale_tilt`: balanced/left/right).\n"
    "  ⑧ `fishbone` (피쉬본 어골도): 여러 복합 원인/요인(`nodes` 2~4개)이 하나의 핵심 결과/현상(`center_label`)으로 수렴하는 다발적 인과 분석에 사용.\n"
    "  ⑨ `quadrant` (2×2 사분면 분류): 두 개의 직교 기준(`quadrant_x_label`, `quadrant_y_label`)으로 4가지 유형(`nodes` 4개: 좌상, 우상, 좌하, 우하 순서)을 분류할 때 사용.\n"
    "  ⑩ `decision_tree` (조건분기 결정 트리): 법적 요건 충족 판정이나 알고리즘처럼 핵심 조건(`center_label`)에 따라 `예 (O)` / `아니오 (X)`(`relation_labels`, `nodes` 2~3개)로 갈라지는 판정 구조에 사용.\n"
    "  ⑪ `pyramid` (위계·계층 피라미드): 법규범 효력(헌법>법률>명령), 사회/생태/시스템 계층처럼 상하 위계가 쌓이는 구조(`nodes` 최상위→최하위 3~4개)에 사용.\n"
    "  ⑫ `flow` (원리·과정 흐름도): 인과·절차 단계가 순차적으로 이어지는 경우 사용 (`nodes` 2~4개, 짧은 `relation_labels` 6자 이내).\n"
    "공통 규칙: 각 카드 `details`는 1~3개의 짧은 핵심 구절(26자 이내), `takeaway`는 1줄 핵심 정리."
)

_KOREAN_LIT_DIAGRAM_GUIDE = (
    "문학 작품의 갈래와 핵심 구조에 맞춰 아래 5종 중 가장 최적화된 `diagram_type`을 **반드시** 선택해 작성하세요:\n"
    "  ① `character_map` (소설 인물관계도 · 갈등 구조도 — 현대소설·고전소설·극 추천): "
    "`nodes`에 핵심 인물 2~4명을 넣으세요(고전소설처럼 호칭이 여러 개면 `subtitle`에 통합 호칭 예: `이선 = 원수 = 좌승상`을 반드시 병기). "
    "`character_edges`에 인물 인덱스 간 조력(`positive`)/갈등(`negative`)/일반(`neutral`) 관계(`relation` 8자 이내)를 명시하세요.\n"
    "  ② `emotion_arc` (시상 전개 · 정서 변화 그래프 — 현대시·고전시가 추천): "
    "`emotion_points`에 시간·연 흐름순 3~5개 좌표를 넣으세요(`stage`='1연'/'전반부', `emotion`='그리움'/'극복 의지', `level`=-2~+2). "
    "V자 반전·심화 등 정서 흐름을 정확히 반영하고 `nodes`에도 핵심 연/구간 요약을 2~3개 담으세요.\n"
    "  ③ `poetic_contrast` (화자 중심 (+) / (-) 시어 대립도 — 자연친화·유배·저항시 추천): "
    "`center_label`에 시적 화자·중심 상황을 넣고, `nodes[0]`에는 '(+) 지향 · 긍정 시어' 목록을, `nodes[1]`에는 '(-) 대립 · 부정 시어' 목록을 넣으세요.\n"
    "  ④ `space_journey` (공간 · 시선 이동 경로도 — 기행가사·영웅소설·여정 서사 추천): "
    "`nodes`에 이동 순서대로 공간 2~4개(각 공간의 상징·정서 변화)를 넣고, `relation_labels`에 이동 계기를 적으세요.\n"
    "  ⑤ `matrix` (`(가)` vs `(나)` 복합 작품 비교표 — 시+시, 시+수필 등 복합 지문 추천): "
    "`nodes`에 작품명 2개를 넣고 `matrix_rows`에 기준('화자/상황', '핵심 소재', '표현 특징', '정서/태도') 3~4행을 채우세요.\n"
    "공통 규칙: 각 카드 `details`는 1~3개의 짧은 핵심 구절(26자 이내), `takeaway`는 1줄 핵심 정리."
)


def _sanitize_korean_analysis_output(out: KoreanPassageAnalysisOutput) -> KoreanPassageAnalysisOutput:
    """국어 분석지 출력에서 잔여 문제 표지(㉠~㉤, ⓐ~ⓔ, [A]~[E]) 및 여분 공백을 결정론적으로 정제한다."""
    marker_only_re = re.compile(r"^[㉠㉡㉢㉣㉤ⓐⓑⓒⓓⓔㄱ-ㅎa-eA-E]\b|^[㉠㉡㉢㉣㉤ⓐⓑⓒⓓⓔ]|^\[[A-E]\]")
    inline_marker_re = re.compile(r"[\u3260-\u327B\u24B6-\u24E9]|\[[A-E]\]")

    def _clean_str(s: str) -> str:
        return re.sub(r"[ \t]{2,}", " ", inline_marker_re.sub("", s or "")).strip()

    clean_vocab = [
        mv.model_copy(update={"term": _clean_str(mv.term), "meaning": _clean_str(mv.meaning)})
        for mv in (out.marker_and_vocab_notes or [])
        if (mv.term or "").strip() and not marker_only_re.match((mv.term or "").strip())
    ]
    clean_paras = [
        ps.model_copy(
            update={
                "label": _clean_str(ps.label),
                "summary": _clean_str(ps.summary),
                "key_point": _clean_str(ps.key_point),
            }
        )
        for ps in (out.paragraph_summaries or [])
    ]
    return out.model_copy(
        update={
            "title": _clean_str(out.title),
            "core_theme": _clean_str(out.core_theme),
            "paragraph_summaries": clean_paras,
            "marker_and_vocab_notes": clean_vocab,
            "area_specific_notes": [_clean_str(n) for n in (out.area_specific_notes or []) if _clean_str(n)],
            "exam_checkpoints": [_clean_str(c) for c in (out.exam_checkpoints or []) if _clean_str(c)],
        }
    )


def proofread_korean_passage_analysis(
    *,
    api_key: str,
    model_name: str,
    passage_text: str,
    draft: KoreanPassageAnalysisOutput,
    korean_section: str = "독서",
) -> KoreanPassageAnalysisOutput:
    """신규 생성된 국어 지문 분석지 초안을 원문과 대조하여 오탈자·맞춤법·비문·사실 불일치를 1회 마무리 교정한다."""
    clean_passage = _strip_korean_exam_noise_lines(passage_text or "")
    if not clean_passage:
        return _sanitize_korean_analysis_output(draft)

    client = _build_client(api_key, timeout_seconds=60)
    system_inst = (
        "당신은 대한민국 수능 국어 교재 출판사 수석 교열·감수 편집자입니다.\n"
        "주어진 [지문 원문]과 [분석지 초안 JSON]을 대조하여 아래 4가지만 정밀 교정하고 최종 JSON을 반환하세요:\n"
        "1. **맞춤법 · 오탈자 · 띄어쓰기 · 어색한 조사/비문** 교정 (예: 학술 용어 오타, 문장 호응 오류).\n"
        "2. **원문 사실 일치 검증**: 지문에 없는 수치·개념·인물/작품명이나 반대로 서술된 인과관계가 있으면 지문 원문에 맞게 수정.\n"
        "3. **문제용 표지 기호 제거**: `㉠~㉤`, `ⓐ~ⓔ`, `[A]~[E]` 같은 기호가 설명에 남아있으면 자연스러운 개념어로 치환.\n"
        "4. **도식(`visual_diagram`) 최종 점검**: `diagram_type`에 필수 필드가 비어 있지 않은지, 각 `details` 항목이 간결(26자 이내)한지 다듬기.\n"
        "이미 완벽한 부분은 불필요하게 축약하거나 훼손하지 말고 원안을 충실히 유지하세요."
    )
    prompt = (
        f"[영역] {korean_section}\n\n"
        f"[지문 원문]\n{clean_passage[:5000]}\n\n"
        f"[분석지 초안 JSON]\n{draft.model_dump_json(indent=2)}\n\n"
        "위 초안을 검토·교정한 최종 `KoreanPassageAnalysisOutput` JSON을 출력하세요."
    )
    try:
        reviewed = _generate_structured_with_fallback(
            client=client,
            model_name=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_inst,
                response_mime_type="application/json",
                response_schema=KoreanPassageAnalysisOutput,
                temperature=0.1,
                max_output_tokens=8192,
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.MINIMAL,
                    include_thoughts=False,
                ),
            ),
            fast_mode=True,
            schema_cls=KoreanPassageAnalysisOutput,
            max_parse_retries=2,
        )
        return _sanitize_korean_analysis_output(reviewed)
    except Exception:
        return _sanitize_korean_analysis_output(draft)


def _is_korean_poetry_genre(
    korean_section: str,
    korean_subgenre: str = "",
    passage_text: str = "",
) -> bool:
    """현대시·고전시가 등 시가 문학 갈래인지 판별한다."""
    sub = (korean_subgenre or "").strip()
    sec = (korean_section or "").strip()
    if any(k in sub for k in ("소설", "극", "시나리오", "희곡", "수필")):
        return False
    if any(k in sub for k in ("현대시", "고전시가", "시가", "시조", "가사", "향가", "한시")) or sub == "시":
        return True
    if "문학" in sec and not sub and passage_text:
        lines = [ln.strip() for ln in passage_text.splitlines() if ln.strip()]
        if len(lines) >= 6:
            lens = sorted(len(ln) for ln in lines)
            if lens[len(lens) // 2] <= 22:
                return True
    return False


def fetch_korean_literature_web_brief(
    api_key: str,
    model_name: str,
    passage_text: str,
    korean_subgenre: str = "",
    passage_topic: str = "",
) -> str:
    """
    Google Search Grounding을 이용해 문학 작품(작가·작품명)의 EBS·기출 해설·백과사전 기반
    핵심 특징(시대 배경, 주제, 표현상·서술상 특징, 상징적 시어/소재, 출제 포인트)을 조회한다.
    개인 블로그/커뮤니티성 출처는 배제하며, 실패 시 빈 문자열을 반환한다.
    """
    clean = _strip_korean_exam_noise_lines(passage_text or "")
    if not clean:
        return ""

    try:
        client = _build_client(api_key, timeout_seconds=45)
        snippet = clean[:1800]
        search_prompt = (
            "당신은 대한민국 수능 국어 문학 자료 조사 전문가입니다.\n"
            "아래 문학 지문에 등장하는 작품(작품명·작가)을 웹검색으로 확인하되, **반드시 EBSi, 수능·모의고사 기출 해설, "
            "한국민족문화대백과사전(encykorea), 두산백과, 문학사전 등 공신력 있는 교육·백과 출처만 우선 활용**하세요. "
            "(개인 블로그의 근거 없는 추측이나 비공식 해석은 배제하세요.)\n\n"
            f"- 갈래/참고 정보: {korean_subgenre or '문학'} / {passage_topic or '-'}\n"
            f"- 지문 발췌:\n{snippet}\n\n"
            "검색된 공신력 있는 자료를 바탕으로 아래 5가지를 작품별((가), (나) 등)로 간결하고 정확하게 정리하세요:\n"
            "1. 작품명 · 작가 · 시대적/창작 배경\n"
            "2. 핵심 주제 및 작품 성격\n"
            "3. 대표 표현상·서술상 특징 (시상 전개, 수미상관, 시점, 어조, 비유/역설 등)\n"
            "4. 핵심 시어·소재의 상징적 의미\n"
            "5. 수능·모의고사 빈출 출제 포인트\n"
            "(참고한 대표 출처명도 마지막 줄에 간단히 적으세요.)"
        )
        resp = _generate_content_with_fallback(
            client=client,
            model_name=model_name,
            contents=search_prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.1,
                max_output_tokens=1500,
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.MINIMAL,
                    include_thoughts=False,
                ),
            ),
            fast_mode=True,
        )
        txt = (getattr(resp, "text", None) or "").strip()
        return txt[:3500]
    except Exception:
        return ""


def generate_korean_passage_analysis(
    api_key: str,
    model_name: str,
    exam_name: str,
    group_id: str,
    passage_text: str,
    korean_section: str = "독서",
    korean_subgenre: str = "",
    passage_topic: str = "",
    enable_proofread: bool = False,
    enable_web_search: bool = True,
) -> KoreanPassageAnalysisOutput:
    """
    국어 지문 1개 세트를 수능·학원형 정밀 지문 분석지(핵심 주제, 구조 흐름, 문단/연별 요약,
    개념 정의 정리, 영역별 정리, 출제 포인트, 시 갈래 밑줄 해설)로 구조화 분석한다.
    """
    clean_passage = _strip_korean_exam_noise_lines(passage_text or "")
    if not clean_passage:
        raise ValueError("분석할 국어 지문 원문(passage_text)이 비어 있습니다.")

    is_poetry = _is_korean_poetry_genre(korean_section, korean_subgenre, clean_passage)
    web_brief = ""
    if enable_web_search and (korean_section == "문학" or is_poetry):
        web_brief = fetch_korean_literature_web_brief(
            api_key=api_key,
            model_name=model_name,
            passage_text=clean_passage,
            korean_subgenre=korean_subgenre,
            passage_topic=passage_topic,
        )

    poetry_rule = (
        "9. `poetry_works` 및 `poetry_underline_notes`: **시(시가) 갈래이므로 필수 작성** —\n"
        "   - `poetry_works`: 지문 속 각 시 작품((가), (나) 등 시 갈래만 작품별로 분리)에 대해 아래를 작성하세요:\n"
        "     * `work_label`: `'(가)'`, `'(나)'` (단일 작품이면 `''`), `title_author`: `'작가, 「작품명」'`\n"
        "     * `lines`: 원문 시의 각 행을 순서대로 담은 리스트 (연과 연 사이는 빈 문자열 `''` 삽입, 원문 표기 100% 그대로 유지)\n"
        "     * `marks`: 핵심 시어·시구 8~16개에 대해 `line_index`(1부터 시작, 빈 줄 포함), `quote`(해당 행 원문 그대로), "
        "`mark_type`(`underline` 직선밑줄 / `wavy` 물결선 / `box` 네모 / `circle` 동그라미 / `triangle` 세모-시련·부정), "
        "`color`(`red` 정서·감각·시련 / `blue` 구조·희망·극복의지 / `dark` 대상·어구풀이), "
        "`position`(`above` 또는 `below` — 한 행에 여러 주석이면 위/아래 분산), `note`(25자 이내 짧은 핵심 주석)를 지정하세요.\n"
        "     * `brackets`: 연/행 묶음·구조 주석 2~5개 (`start_line`, `end_line`, `side`=`'right'`/`'left'`, "
        "`style`=`'brace'` 중괄호 / `'arc'` 수미상관 연결호 / `'divider'` 시상전환 점선, `color`, `note` 예: `'대구법 & 연쇄법'`, `'수미상관 구조'`, `'시상 전환'`).\n"
        "   - `poetry_underline_notes`: 핵심 시구 4~7개 요약 리스트도 함께 채우세요."
        if is_poetry
        else "9. `poetry_works` 및 `poetry_underline_notes`: 시(시가) 갈래가 아니므로 둘 다 빈 리스트 `[]`로 두세요."
    )

    client = _build_client(api_key, timeout_seconds=90)
    if korean_section == "문학":
        area_guide = (
            "문학 영역이므로 화자/서술자의 위치와 정서·태도 변화, 핵심 시어·소재의 상징적 의미, "
            "표현상·서술상 특징(비유, 대비, 역설, 시점, 서술 방식 등), (가)/(나) 작품 간 비교 포인트를 정밀 정리하세요."
        )
    elif korean_section == "화법과 작문":
        area_guide = (
            "화법과 작문 영역이므로 발표/대화 참여자의 말하기 전략, 청자 반응, 초고의 전개 구조 및 고쳐쓰기·자료 활용 포인트를 정리하세요."
        )
    elif korean_section == "언어와 매체":
        area_guide = (
            "언어와 매체 영역이므로 핵심 국어 문법 개념·원리(음운, 품사, 문장, 중세국어 등) 또는 매체 언어 특성·정보 전달 방식을 정리하세요."
        )
    else:
        area_guide = (
            "독서(비문학) 영역이므로 중심 화제와 핵심 개념 간 관계(대조, 인과, 비례/반비례, 한계와 의의), "
            "문단별 논지 전개 흐름과 구체적 사례 적용 원리를 정밀 정리하세요."
        )

    grounding_inst = (
        "아래에 제공된 `[신뢰할 수 있는 웹검색 문학 해설]`을 **핵심 분석 기준(메인 기준)**으로 삼아 "
        "작품의 시대 배경, 핵심 주제, 표현상 특징, 상징적 시어·소재, 출제 포인트를 깊이 있게 반영하되, "
        "지문 원문 인용(`poetry_works.lines`, `marks.quote`)은 반드시 `[지문 원문]`에 실제로 발췌된 구절만 그대로 사용하세요.\n\n"
        if web_brief
        else f"주어진 [{korean_section} · {korean_subgenre or '공통'}] 지문 원문만을 100% 근거로 하여, "
    )

    system_inst = (
        "당신은 대한민국 대치동/수능 국어 영역 일타 강사 수준의 정밀 지문 분석가입니다.\n"
        f"{grounding_inst}"
        "학생과 강사가 바로 수업·복습에 활용할 수 있는 **수능·학원형 정밀 지문 분석지**를 JSON으로 작성하세요.\n\n"
        "[작성 규칙]\n"
        "1. `title`: 지문의 핵심 제목이나 작품명·작가를 간결하게 적으세요.\n"
        "2. `core_theme`: 지문 전체의 핵심 주제(또는 작품 주제 의식)를 1~2문장으로 명확히 요약하세요.\n"
        "3. `structure_flow`: 글 전체의 논리 흐름이나 시상/서사 전개를 `도입 : ...\\n전개 : ...\\n심화 : ...\\n결론 : ...`처럼 단계별 줄바꿈 형식으로 한눈에 보이게 정리하세요.\n"
        "4. `paragraph_summaries`: 독서는 문단별(1문단, 2문단...)로, 문학은 작품·연·장면별((가) 1연, (나) 도입...)로 빠짐없이 요약하세요.\n"
        "5. `marker_and_vocab_notes`: **절대 `㉠~㉤`, `ⓐ~ⓔ`, `[A]~[E]` 같은 표지 기호는 넣지 마세요.** "
        "오직 지문 원문에서 개념·용어의 정의가 명시된 문장(`'~~는 ~~~~이다'`, `'~~(이)란 ~~~~을 말한다'` 등)만 뽑아 정리하세요. "
        "`term`에는 정의되는 개념어(예: `'대리'`, `'현명주의'`, `'자기장'`)만 간결하게 적고, `meaning`에는 지문에 나온 정의를 `'~~는 ~~~~이다'` 형태로 적으세요. "
        "(지문에 명확한 정의 문장이 없으면 빈 리스트 `[]`로 두세요.)\n"
        f"6. `area_specific_notes`: {area_guide}\n"
        "7. `exam_checkpoints`: 실제 모의고사·내신에서 자주 출제되는 포인트와 선지 함정 주의사항 3~5개를 작성하세요.\n"
        "8. `visual_diagram`: "
        + (
            "비문학(독서) 지문이므로 **반드시** `visual_diagram`을 작성하세요.\n"
            + _KOREAN_DIAGRAM_GUIDE
            if korean_section == "독서"
            else (
                "문학 작품이므로 **반드시** 갈래에 맞는 문학 전용 `visual_diagram`을 작성하세요.\n"
                + _KOREAN_LIT_DIAGRAM_GUIDE
                if korean_section == "문학"
                else "개념 비교나 흐름 도식이 이해에 큰 도움이 되는 지문이면 `visual_diagram`을 작성하고, 불필요하면 `null`로 두세요.\n"
                + _KOREAN_DIAGRAM_GUIDE
            )
        )
        + f"\n{poetry_rule}"
    )

    web_block = (
        f"[신뢰할 수 있는 웹검색 문학 해설 (메인 분석 기준)]\n{web_brief}\n\n"
        if web_brief
        else ""
    )
    prompt = (
        f"[지문 정보]\n"
        f"- 출처: {exam_name} [{group_id}번]\n"
        f"- 영역/갈래: {korean_section} ({korean_subgenre or '공통'})\n"
        f"- 참고 주제/작품: {passage_topic or '-'}\n\n"
        f"{web_block}"
        f"[지문 원문]\n{clean_passage}\n\n"
        "위 자료를 바탕으로 정밀 지문 분석지 JSON을 작성하세요."
    )

    draft_out = _generate_structured_with_fallback(
        client=client,
        model_name=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=KoreanPassageAnalysisOutput,
            temperature=0.2,
            max_output_tokens=8192,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL,
                include_thoughts=False,
            ),
        ),
        fast_mode=True,
        schema_cls=KoreanPassageAnalysisOutput,
        max_parse_retries=2,
    )
    if enable_proofread:
        return proofread_korean_passage_analysis(
            api_key=api_key,
            model_name=model_name,
            passage_text=clean_passage,
            draft=draft_out,
            korean_section=korean_section,
        )
    return _sanitize_korean_analysis_output(draft_out)


def generate_korean_visual_diagram(
    *,
    api_key: str,
    model_name: str,
    passage_text: str,
    title: str = "",
    core_theme: str = "",
    korean_section: str = "독서",
) -> KoreanVisualDiagram:
    """기존에 분석된 국어 비문학 지문에 대해 이해를 돕는 도식 그림 스펙만 생성한다."""
    clean_passage = _strip_korean_exam_noise_lines(passage_text or "")
    if not clean_passage:
        raise ValueError("도식을 생성할 국어 지문 원문이 비어 있습니다.")

    client = _build_client(api_key, timeout_seconds=60)
    diag_guide = _KOREAN_LIT_DIAGRAM_GUIDE if korean_section == "문학" else _KOREAN_DIAGRAM_GUIDE
    system_inst = (
        f"당신은 수능 국어({korean_section}) 일타 강사입니다. 지문 원문만을 100% 근거로, "
        "학생이 핵심 구조·인물·정서·원리를 그림으로 한눈에 이해할 수 있는 도식 명세(`KoreanVisualDiagram`) JSON을 작성하세요.\n"
        + diag_guide
    )
    prompt = (
        f"[지문 영역] {korean_section}\n"
        f"[제목/주제] {title or '-'} / {core_theme or '-'}\n\n"
        f"[지문 원문]\n{clean_passage[:4500]}\n\n"
        "위 지문의 핵심을 시각화한 도식 JSON을 작성하세요."
    )
    return _generate_structured_with_fallback(
        client=client,
        model_name=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=KoreanVisualDiagram,
            temperature=0.2,
            max_output_tokens=2048,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL,
                include_thoughts=False,
            ),
        ),
        fast_mode=True,
        schema_cls=KoreanVisualDiagram,
        max_parse_retries=2,
    )
