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
    _parse_structured,
    optimize_image_for_vision,
    prepare_pdf_part,
)
from src.models import (
    BatchKoreanExamExtractionOutput,
    KoreanPassageAnalysisOutput,
    KoreanQuestionExtractionItem,
    KoreanSetReconstructionOutput,
    KoreanSingleExtractionOutput,
    KoreanVisualDiagram,
    QuestionRecord,
    RewrittenQuestion,
    normalize_korean_group_id,
)
from src.text_formatter import format_explanation_text


def _sync_korean_group_passages(items: list[KoreanQuestionExtractionItem]) -> None:
    """동일한 passage_group_id를 공유하는 국어 세트 문항들의 지문·주제·세부 갈래를 동기화한다."""
    best_passage: dict[str, str] = {}
    best_topic: dict[str, str] = {}
    best_subgenre: dict[str, str] = {}
    for item in items:
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

    resp = _generate_content_with_fallback(
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
    )
    parsed = _parse_structured(resp, BatchKoreanExamExtractionOutput)

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

    resp = _generate_content_with_fallback(
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
    )
    out = _parse_structured(resp, KoreanSingleExtractionOutput)
    out.answer = _normalize_extracted_answer(out.answer)
    out.passage_group_id = normalize_korean_group_id(out.passage_group_id, q_num)
    if not (out.passage_text or "").strip():
        out.passage_text = (shared_passage_hint or question_text_hint or "").strip()
    _apply_lang_standard_validation(out, "국어")
    return out


def generate_korean_set_reconstruction(
    api_key: str,
    model_name: str,
    source_records: list[QuestionRecord],
    passage_text: str,
    korean_section: str = "독서",
    korean_subgenre: str = "",
    passage_topic: str = "",
) -> list[tuple[QuestionRecord, RewrittenQuestion]]:
    """
    국어 지문 1개 세트(원본 문항 수 N개, 보통 3~6문항)를 받아
    지문 전체에서 핵심 출제 포인트를 자유 유형으로 새로 설계하여 N개의 수능형 국어 문항 세트를 일괄 생성한다.
    반환되는 각 (QuestionRecord, RewrittenQuestion) 쌍 중 첫 문항의 rewritten_stem 상단에
    세트 공통 헤더 및 지문 원문을 배치하여 모드 A(PDF)·모드 B(웹 풀기)에서 1회만 출력되도록 구성한다.
    """
    ordered_recs = sorted(source_records, key=lambda r: r.q_num)
    if not ordered_recs:
        return []

    n_questions = len(ordered_recs)
    clean_passage = (passage_text or "").strip()
    if not clean_passage:
        raise ValueError("국어 지문 원문(passage_text)이 비어 있어 세트 문항을 재구성할 수 없습니다.")

    client = _build_client(api_key, timeout_seconds=120)
    orig_summary_lines = [
        f"- 원본 {r.q_num}번 ({r.unit_name or '기출'}): 핵심={'; '.join(r.core_principles[:2])} | 정답={r.answer}"
        for r in ordered_recs
    ]
    orig_summary_block = "\n".join(orig_summary_lines)

    if korean_section == "문학":
        type_guide = (
            "문학 영역 추천 유형을 지문 특성에 맞게 자유롭게 배분하세요: "
            "표현상·서술상 특징, 시적 화자/서술자 및 인물의 정서·태도, 구절·소재(㉠~㉤/ⓐ~ⓔ 등)의 함축적 의미, "
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
            "<보기>의 구체적 사례/상황에 지문 원리 적용 추론, 특정 문맥 어휘/구절(ⓐ~ⓔ, ㉠~㉤)의 문맥적 의미."
        )

    system_inst = (
        "당신은 대한민국 수능·평가원 국어 영역 최고 수준 문항 출제위원입니다.\n"
        f"주어진 [{korean_section} · {korean_subgenre or '공통'}] 지문을 바탕으로, 기존 기출 문항을 그대로 베끼지 말고 "
        f"**총 {n_questions}개의 고품질 수능 국어 세트 문항**을 자유 유형으로 새롭게 창작하세요.\n\n"
        "[출제 절대 원칙]\n"
        "1. **지문 100% 근거 원칙**: 모든 문항과 선지의 정오 판정은 반드시 아래 [공통 지문 원문]에 직접 명시되거나 논리적으로 도출 가능한 사실에만 근거해야 합니다. 지문에 없는 외부 지식을 정답 근거로 쓰지 마세요.\n"
        "2. **지문 표지 정밀 연동**: 지문 원문에 `[A]`, `(가)`, `(나)`, `㉠~㉤`, `ⓐ~ⓔ` 같은 표지가 있으면 발문이나 선지에서 해당 표지를 정확히 참조하세요. 지문에 없는 표지(예: 지문에 `㉮`가 없는데 `㉮`를 묻기)를 절대 지어내지 마세요.\n"
        f"3. **자유 유형 균형 출제 ({n_questions}문항)**: {type_guide}\n"
        "   - 최소 1문항 이상은 구체적인 `<보기>` 박스를 포함한 고난도 적용·감상 문항으로 출제하세요.\n"
        "4. **발문·<보기> 형식**: `stem`에는 각 문항의 발문(예: `윗글에 대한 이해로 가장 적절한 것은?`)과 필요한 경우 `<보기>\\n...` 박스만 작성하세요. **공통 지문 본문은 `stem`에 반복해서 넣지 마세요.**\n"
        "5. **선지 5개(`choices`) & 정답 분산**: `choices`는 선지 번호(①~⑤)를 제외한 5개 문자열 리스트로 작성하고, `answer`는 `'1'`~`'5'` 중 단일 숫자 문자열로 적되 정답 번호가 한 번호에 몰리지 않게 골고루 배치하세요.\n"
        "6. **정교한 해설(`explanation`)**: 정답 선지가 왜 지문과 일치하는지, 그리고 나머지 4개 오답 선지가 지문의 어떤 부분과 어긋나는지(인과 도치, 주체 오류, 범위 과장 등) 선지별로 명확히 설명하세요."
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

    resp = _generate_content_with_fallback(
        client=client,
        model_name=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_inst,
            response_mime_type="application/json",
            response_schema=KoreanSetReconstructionOutput,
            temperature=0.35,
            max_output_tokens=16384,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL,
                include_thoughts=False,
            ),
        ),
        fast_mode=True,
    )
    parsed = _parse_structured(resp, KoreanSetReconstructionOutput)
    gen_qs = sorted(parsed.questions or [], key=lambda x: x.slot_index)[:n_questions]
    if not gen_qs:
        raise ValueError("AI가 국어 세트 문항을 생성하지 못했습니다. 다시 시도해 주세요.")

    set_intro_txt = (parsed.set_intro or "").strip() or f"[1~{len(gen_qs)}] 다음 글을 읽고 물음에 답하시오."
    results: list[tuple[QuestionRecord, RewrittenQuestion]] = []
    for idx, gq in enumerate(gen_qs):
        base_rec = ordered_recs[idx] if idx < len(ordered_recs) else ordered_recs[-1]
        clean_choices = [re.sub(r"^[①②③④⑤]\s*", "", (c or "").strip()) for c in (gq.choices or []) if (c or "").strip()]
        while len(clean_choices) < 5:
            clean_choices.append("(선지 보완 필요)")
        clean_choices = clean_choices[:5]

        clean_stem = (gq.stem or "").strip()
        if idx == 0:
            full_stem = f"{set_intro_txt}\n\n{clean_passage}\n\n{clean_stem}"
        else:
            full_stem = clean_stem

        norm_ans = _normalize_extracted_answer(gq.answer)
        rw = RewrittenQuestion(
            original_question_id=base_rec.id,
            rewritten_stem=full_stem,
            rewritten_choices=clean_choices,
            replaced_terms_map=[],
            pdf_grounded_explanation=format_explanation_text(gq.explanation, for_markdown=False),
            preserved_answer=norm_ans,
            self_verification_note=f"국어 {korean_section} 지문 세트 자유 유형 재구성 완료 ({gq.question_type_label})",
            visual_term_note="",
            target_question_type=gq.question_type_label.strip() or "국어 세트 재구성",
            subject_domain="국어",
        )
        results.append((base_rec, rw))

    return results


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
        resp = _generate_content_with_fallback(
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
        )
        reviewed = _parse_structured(resp, KoreanPassageAnalysisOutput)
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

    resp = _generate_content_with_fallback(
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
    )
    draft_out = _parse_structured(resp, KoreanPassageAnalysisOutput)
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
    resp = _generate_content_with_fallback(
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
    )
    return _parse_structured(resp, KoreanVisualDiagram)
