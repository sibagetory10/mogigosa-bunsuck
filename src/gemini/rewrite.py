from __future__ import annotations

import re
from typing import Any

from google.genai import types

from src.gemini.core import (
    ForbiddenTermRemainingError,
    _build_client,
    _generate_content_with_fallback,
    _generate_structured_with_fallback,
    _parse_structured,
    optimize_image_for_vision,
)
from src.models import ConceptWhitelist, QuestionRecord, RewrittenQuestion
from src.text_formatter import (
    find_forbidden_terms_in_text,
    format_chemical_scripts,
    format_explanation_text,
    normalize_exam_layout_text,
)


def rewrite_question_for_pdf(
    api_key: str,
    model_name: str,
    question: QuestionRecord,
    crop_png_bytes: bytes | None,
    whitelist: ConceptWhitelist,
    out_of_scope_terms: list[str],
    sliced_text: str = "",
    previous_attempt: RewrittenQuestion | None = None,
    extra_feedback: str = "",
) -> RewrittenQuestion:
    """
    풀이 원리는 PDF 범위 안이지만 상위 과목 용어(out_of_scope_terms)가 포함된 문항을
    업로드된 PDF 허용 용어만 사용해 개조하고, 금지 용어 잔존 여부를 후처리 정규식으로 검증한다.
    잔존 시 누락 용어를 명시해 1회 자동 재개조하며, 재개조 후에도 잔존 시 ForbiddenTermRemainingError를 발생시킨다.
    """
    client = _build_client(api_key)

    system_inst = (
        "당신은 고난도 교육과정 기반 문항 리라이터(PDF-Grounded Rewriter)입니다.\n"
        "다음 절대 원칙을 반드시 지키세요:\n"
        "1. 업로드된 개념서 PDF에 있는 개념(pdf_concepts)과 용어(pdf_vocabulary), 그리고 일반 일상 한국어만 사용하세요.\n"
        "2. 범위 외 상위 용어 목록(out_of_scope_terms) 및 개념서에 없는 상위 단위/용어(`M`, `몰 농도`, `몰농도`, `mol`, `몰수`, `분자량`, `화학식량`, `엔탈피` 등)는 개조된 지문, <자료>, <보기>, 선지 어디에도 절대 남기지 마세요.\n"
        "3. **핵심 원리 억지 동어반복 치환 금지**: 개념서 범위를 벗어나는 실험/원리 개념 자체를 동어반복 문장으로 어색하게 풀어쓰지 말고, 개념서 허용 용어와 구체적 수치·입자 수 비율로 논리가 온전히 성립되게 표현하세요.\n"
        "4. **자연스러운 국문 문장 구조 유지**: 용어 치환 후 문장 끝이 어색하게 끊기지 않도록(예: `'질량: x g x는?'` 금지 → `'질량을 x g이라 할 때, x는?'`) 조사와 어미를 매끄러운 시험지 문체로 다듬으세요.\n"
        "5. **구조화 줄바꿈 (필수)**: 한 문단으로 뭉치지 않도록 도입 발문 / `<자료>`·`<조건>`·`<실험 과정>` / 질문 문장 / `<보기>` / `(단, ...)` 사이는 빈 줄(`\\n\\n`)로 띄우고, 자료 안의 각 항목(`◦`, `(가)`, `(나)`)과 `<보기>`의 `ㄱ.`, `ㄴ.`, `ㄷ.` 및 `pdf_grounded_explanation`의 풀이 단계(`1.`, `2.`, `ㄱ.`, `ㄴ.`)는 반드시 각각 줄바꿈(`\\n`)해 별도의 줄에 적으세요.\n"
        "6. **그림·그래프 시각 자료 안내(`visual_term_note`)**: 원본 그림·표·그래프의 축이나 범례에 상위 단위(예: `mol`, `M`)가 그려져 있다면, `visual_term_note`에 예: `그림 속 'mol'은 본문의 '입자 수'와 같은 의미로 해석합니다`처럼 학생용 안내 문구를 작성하세요(시각 자료에 상위 용어가 없으면 빈 문자열 `\"\"`).\n"
        "7. 문제의 수학적·논리적 구조, 수치 조건, 정답의 참/거짓 판정 로직은 원본과 **100% 동일하게** 유지하며 `preserved_answer`도 원본 정답과 동일해야 합니다.\n"
        "8. `rewritten_choices`는 객관식이면 5개 항목 리스트(선지 번호 ①~⑤ 제외하고 내용만), 주관식이면 빈 리스트 []로 반환하세요."
    )

    q_text = (getattr(question, "question_text", "") or "").strip()
    has_visual_cue = any(
        kw in q_text
        for kw in ("그림", "그래프", "표", "도표", "모식도", "사진", "지도", "회로", "실험 장치")
    )
    need_image = bool(crop_png_bytes) and (len(q_text) < 40 or has_visual_cue)

    context_excerpt = sliced_text[:6000] if sliced_text else ""
    q_text_block = f"[원본 문항 텍스트]\n{q_text}\n\n" if q_text else ""

    def _invoke_rewrite(correction_note: str = "") -> RewrittenQuestion:
        prev_block = ""
        if previous_attempt is not None:
            prev_block = (
                "\n[직전 개조 결과 참고 (개선·재변환 필요)]\n"
                f"- 직전 지문: {previous_attempt.rewritten_stem}\n"
                f"- 직전 선지: {previous_attempt.rewritten_choices}\n"
            )
        feedback_block = f"\n[재변환 강화 지시]\n{correction_note}\n" if correction_note else ""

        prompt = (
            f"[원본 문항 정보]\n"
            f"- ID: {question.id} ({question.exam_name} {question.q_num}번)\n"
            f"- 본질 원리: {question.core_principles}\n"
            f"- 원본 정답: {question.answer}\n"
            f"- 원본 공식 해설 요약: {question.official_explanation}\n"
            f"- 반드시 제거·치환해야 할 범위 외 상위 용어: {out_of_scope_terms}\n\n"
            f"{q_text_block}"
            f"{prev_block}"
            f"{feedback_block}"
            f"[업로드된 PDF 허용 개념 (pdf_concepts)]\n{whitelist.pdf_concepts}\n\n"
            f"[업로드된 PDF 허용 용어 (pdf_vocabulary)]\n{whitelist.pdf_vocabulary}\n\n"
            f"[업로드된 PDF 본문 참고 발췌]\n{context_excerpt}\n\n"
            "위 원본 문항을 보고, 범위 외 상위 용어를 PDF 내부 표현으로 완전히 치환한 '변형 문제 카드'와 'PDF 맞춤 해설'을 작성하세요."
        )

        contents: list[Any] = []
        if need_image and crop_png_bytes:
            contents.append(types.Part.from_bytes(data=optimize_image_for_vision(crop_png_bytes), mime_type="image/png"))
        contents.append(prompt)

        rw = _generate_structured_with_fallback(
            client=client,
            model_name=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_inst,
                response_mime_type="application/json",
                response_schema=RewrittenQuestion,
                temperature=0.2,
            ),
            fast_mode=True,
            schema_cls=RewrittenQuestion,
            max_parse_retries=2,
        )
        rw.preserved_answer = question.answer
        rw.rewritten_stem = normalize_exam_layout_text(rw.rewritten_stem)
        rw.rewritten_choices = [format_chemical_scripts(c.strip()) for c in (rw.rewritten_choices or []) if c.strip()]
        rw.pdf_grounded_explanation = format_explanation_text(rw.pdf_grounded_explanation, for_markdown=False)
        return rw

    initial_note = extra_feedback.strip()
    if previous_attempt is not None:
        prev_residuals = find_forbidden_terms_in_text(
            [previous_attempt.rewritten_stem, *previous_attempt.rewritten_choices],
            out_of_scope_terms=out_of_scope_terms,
            allowed_vocab=whitelist.pdf_vocabulary,
        )
        if prev_residuals:
            add_msg = (
                f"이전 변환 결과에 아직 상위 용어 {prev_residuals}가 남아 있습니다. "
                "개념서 허용 용어와 구체적 수치 환산으로 100% 완전히 치환하고, "
                "어색한 조사·문장 끊김도 자연스러운 국문으로 다시 다듬으세요."
            )
            initial_note = f"{initial_note}\n{add_msg}".strip()
        elif not initial_note:
            initial_note = "이전 변환 결과의 문장을 더 자연스러운 국문 시험지 문체로 다듬고, 개념서 허용 용어만 사용해 완전히 재변환하세요."

    rewritten = _invoke_rewrite(correction_note=initial_note)
    residual_terms = find_forbidden_terms_in_text(
        [rewritten.rewritten_stem, *rewritten.rewritten_choices],
        out_of_scope_terms=out_of_scope_terms,
        allowed_vocab=whitelist.pdf_vocabulary,
    )

    if residual_terms:
        retry_note = (
            f"후처리 검증 결과 개조 지문/선지에 범위 외 금지 용어 {residual_terms}가 여전히 남아 있습니다! "
            "해당 용어나 단위를 발문, <자료>, <보기>, 선지에서 단 한 글자도 남기지 말고 "
            "개념서 허용 용어(pdf_vocabulary) 및 구체적 수치/입자 수 표현으로 100% 치환해 다시 작성하세요."
        )
        rewritten = _invoke_rewrite(correction_note=retry_note)
        residual_after_retry = find_forbidden_terms_in_text(
            [rewritten.rewritten_stem, *rewritten.rewritten_choices],
            out_of_scope_terms=out_of_scope_terms,
            allowed_vocab=whitelist.pdf_vocabulary,
        )
        if residual_after_retry:
            raise ForbiddenTermRemainingError(
                f"재개조 후에도 범위 외 금지 용어({', '.join(residual_after_retry)})가 잔존하여 문항을 제외했습니다."
            )

    return rewritten


def generate_pdf_grounded_explanation(
    api_key: str,
    model_name: str,
    question: QuestionRecord,
    crop_png_bytes: bytes | None,
    whitelist: ConceptWhitelist,
    user_answer: str,
    rewritten: RewrittenQuestion | None = None,
    sliced_text: str = "",
    concept_reference: str = "",
) -> str:
    """모드 B: 문항 이미지 + PDF 범위 + 사용자 오답을 바탕으로 PDF 용어만 사용한 AI 자동 해설을 생성한다."""
    client = _build_client(api_key)

    system_inst = (
        "당신은 학생 맞춤형 1:1 AI 튜터입니다.\n"
        "반드시 **선택된 개념서/교과서 범위의 개념(pdf_concepts)과 허용 용어(pdf_vocabulary)만** 사용하여 설명하세요. "
        "개념서 범위를 벗어나는 상위 교육과정 용어(예: 학생이 아직 안 배운 전공 용어)는 절대 사용하지 마세요.\n"
        "학생이 고른 답과 정답을 비교해 왜 맞았는지 또는 어디서 헷갈렸는지 짚어주고, 단계별 풀이를 친절하고 명료하게 제시하세요."
    )

    rewritten_block = ""
    if rewritten is not None:
        choices_str = "\n".join(
            f"{idx + 1}) {c}" for idx, c in enumerate(rewritten.rewritten_choices)
        )
        rewritten_block = (
            f"\n[PDF 맞춤 개조 문항]\n{rewritten.rewritten_stem}\n{choices_str}\n"
        )

    ref_line = f"[참고 개념서 단원 및 페이지] {concept_reference}\n" if concept_reference else ""
    prompt = (
        f"[문항 정보] {question.exam_name} {question.q_num}번 ({question.curriculum_subject})\n"
        f"[정답] {question.answer}\n"
        f"[학생이 입력한 답] {user_answer or '(미입력)'}\n"
        f"{ref_line}"
        f"{rewritten_block}\n"
        f"[공식 해설 참고] {question.official_explanation}\n\n"
        f"[반드시 이 범위 내 개념·용어만 사용]\n"
        f"- 허용 개념: {whitelist.pdf_concepts}\n"
        f"- 허용 용어: {whitelist.pdf_vocabulary}\n"
        f"- 개념서 본문 일부: {sliced_text[:4000]}\n\n"
        "학생에게 보여줄 단계별 AI 자동 해설을 마크다운으로 작성해 주세요."
    )

    contents: list[Any] = []
    if crop_png_bytes:
        contents.append(types.Part.from_bytes(data=optimize_image_for_vision(crop_png_bytes), mime_type="image/png"))
    contents.append(prompt)

    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            temperature=0.2,
        ),
        fast_mode=True,
    )
    out_md = (resp.text or "").strip()
    if concept_reference and concept_reference not in out_md:
        out_md = f"{out_md}\n\n---\n**{concept_reference}**"
    return out_md


def chat_followup_with_pdf(
    api_key: str,
    model_name: str,
    question: QuestionRecord,
    crop_png_bytes: bytes | None,
    whitelist: ConceptWhitelist,
    chat_history: list[dict[str, str]],
    user_question: str,
    rewritten: RewrittenQuestion | None = None,
    sliced_text: str = "",
) -> str:
    """모드 B: 문항별 추가 질문 채팅에 대해 PDF 개념/용어 범위 내에서만 답변한다."""
    client = _build_client(api_key)

    system_inst = (
        "당신은 업로드된 개념서 PDF 범위 안에서만 답변하는 친절한 학습 튜터입니다. "
        "반드시 아래 허용 개념(pdf_concepts)과 허용 용어(pdf_vocabulary)만 사용해 학생의 추가 질문에 답하세요. "
        "PDF에 없는 상위 개념을 묻는 질문이라면 '현재 업로드하신 개념서 범위 밖의 용어이므로 개념서 내 표현으로 풀어서 설명드리면~'처럼 "
        "PDF 범위 안의 표현으로 치환해 안내하세요."
    )

    rewritten_block = ""
    if rewritten is not None:
        choices_str = "\n".join(
            f"{idx + 1}) {c}" for idx, c in enumerate(rewritten.rewritten_choices)
        )
        rewritten_block = f"\n[개조 문제]\n{rewritten.rewritten_stem}\n{choices_str}\n"

    history_text = "\n".join(
        f"{'학생' if m['role'] == 'user' else 'AI 튜터'}: {m['content']}"
        for m in chat_history[-8:]
    )

    prompt = (
        f"[문항] {question.exam_name} {question.q_num}번 | 정답: {question.answer}\n"
        f"{rewritten_block}\n"
        f"[허용 개념] {whitelist.pdf_concepts}\n"
        f"[허용 용어] {whitelist.pdf_vocabulary}\n"
        f"[개념서 참고] {sliced_text[:3000]}\n\n"
        f"[이전 대화]\n{history_text}\n\n"
        f"[학생의 새 질문]\n{user_question}"
    )

    contents: list[Any] = []
    if crop_png_bytes:
        contents.append(types.Part.from_bytes(data=optimize_image_for_vision(crop_png_bytes), mime_type="image/png"))
    contents.append(prompt)

    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            temperature=0.25,
        ),
        fast_mode=True,
    )
    return (resp.text or "").strip()
