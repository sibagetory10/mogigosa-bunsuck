from __future__ import annotations

import random
import re
from typing import Any, Sequence

from src.gemini.english_validators.transform_helpers import (
    _AI_META_MONOLOGUE_RE,
    _COMMON_ENG_VERBS,
    _DANGLING_TO_END_RE,
    _DETERMINER_SET,
    _ORDER_SEQ_RE,
    _REPEATABLE_PREPS,
    _SENT_ABBREV_RE,
    _build_english_transform_rule,
    _count_english_sentences,
    _ensure_korean_instruction_prefix,
    _extract_given_sentence_from_insertion_stem,
    _has_order_scrambled_sentences,
    _is_irrelevant_answer_from_source,
    _is_order_abc_canonical,
    _normalize_pdf_hyphens,
    _preprocess_english_source_passage,
    _rebuild_insertion_stem_from_source,
    _reconcile_english_insertion_answer,
    _reconcile_english_order_answer,
    _reconcile_english_summary_answer,
    _sentence_has_english_verb,
    _split_english_sentences,
    _strip_stray_exam_qnum_noise,
)
from src.gemini.usage import _CIRCLED_DIGIT_TO_NUM, _normalize_extracted_answer
from src.models import RewrittenQuestion
from src.text_formatter import normalize_english_exam_text, strip_english_embedded_choices

def _validate_english_transformed_output(
    rw: RewrittenQuestion,
    target_type: str,
    vocab_style: str,
    source_passage: str = "",
) -> list[str]:
    """생성된 영어 변형 문항의 필수 태그·메타 독백·어휘 실제 교체·미완성 문장 여부를 검증한다."""
    issues: list[str] = []
    stem = rw.rewritten_stem or ""
    t = (target_type or "").strip()

    # 1. AI 영문/한글 독백(메타 발화) 혼입 차단
    all_text_fields = [stem, *(rw.rewritten_choices or []), rw.pdf_grounded_explanation or ""]
    for fld in all_text_fields:
        m_meta = _AI_META_MONOLOGUE_RE.search(fld or "")
        if m_meta:
            issues.append(
                f"AI 내부 사고/자기 수정 메타 문구({m_meta.group(0)!r})가 출력에 혼입되었습니다. 메타 독백 없이 순수 문제와 해설만 반환하세요."
            )
            break

    # 1-b. 영문 지문 본문(발문·섹션 헤더·각주 제외)에 한글 코멘트가 섞였는지 차단
    for ln_raw in stem.splitlines():
        ln_s = ln_raw.strip()
        if not ln_s:
            continue
        if re.search(r"가장\s*적절한\s*(?:것은|곳은)\??$", ln_s):
            continue
        if re.match(r"^\[(?:주어진\s*글|주어진\s*문장|요약문)\]$", ln_s):
            continue
        if ln_s.startswith("*"):
            continue
        if re.search(r"[가-힣]{2,}", ln_s) and re.search(r"[A-Za-z]{4,}", ln_s):
            issues.append(
                f"영문 지문 본문 안에 한글 설명/메모({ln_s[:40]!r})가 섞였습니다. 지문 본문은 순수 영어만 남기세요."
            )
            break

    # 4. 문장 끝 단어 탈락 방지 ('has to.', 'able to.' 등)
    m_to = _DANGLING_TO_END_RE.search(stem)
    if m_to:
        issues.append(
            f"지문 내에 부정사 뒤 동사원형이 탈락한 미완성 문장({m_to.group(0)!r})이 감지되었습니다. 빠진 단어를 복원해 완전한 문장으로 작성하세요."
        )

    ans = _normalize_extracted_answer(rw.preserved_answer)
    if ans not in ("1", "2", "3", "4", "5"):
        issues.append(f"정답(preserved_answer)이 '1'~'5' 중 하나여야 합니다 (현재: {rw.preserved_answer!r}).")

    if len(rw.rewritten_choices or []) != 5:
        issues.append(f"선택지(rewritten_choices)는 정확히 5개여야 합니다 (현재: {len(rw.rewritten_choices or [])}개).")

    u_matches = re.findall(r"<u\b[^>]*>(.*?)</u>", stem, flags=re.I | re.S)
    u_count = len(u_matches)
    if "어휘" in t and source_passage:
        src_cnt = _count_english_sentences(source_passage)
        gen_cnt = _count_english_sentences(stem)
        if src_cnt >= 3 and gen_cnt != src_cnt:
            issues.append(
                f"문맥상 어휘 유형은 원문 문장 수({src_cnt}개)와 동일해야 합니다 (현재 {gen_cnt}문장). "
                "지문 끝에 새 문장을 추가하거나 기존 문장을 삭제하지 말고 원문 본문 단어만 교체하세요."
            )

    if "어법" in t or ("어휘" in t and "선택형" not in vocab_style and "(A)" not in vocab_style):
        missing_markers = [mk for mk in ("①", "②", "③", "④", "⑤") if stem.count(mk) != 1]
        if u_count != 5 or missing_markers:
            issues.append(
                f"지문(rewritten_stem) 안에 `① <u>...</u>` ~ `⑤ <u>...</u>` 밑줄 5개가 모두 정확히 1개씩 포함되어야 합니다 "
                f"(현재 `<u>` {u_count}개, 누락/중복 번호: {missing_markers})."
            )
        elif "어법" in t:
            exp_txt = rw.pdf_grounded_explanation or ""
            if re.search(r"(?:의미상|의미적|문맥상\s*반대|반의어|부정어\s*(?:not|no)\s*(?:이|가)?\s*빠)", exp_txt):
                issues.append(
                    "어법 문제 해설에 어휘 뜻풀이('의미상'/'의미적'/'반의어'/'부정어 누락')가 포함되었습니다. 어휘 문제와 혼동하지 말고 명백한 문법 비문 1개만 출제하세요."
                )
            elif re.search(r"(?:과거형\s*대신\s*현재형|현재형\s*대신\s*과거형|시제\s*흐름과\s*일치)", exp_txt):
                issues.append(
                    "어법 문제에서 단순 과거/현재 시제 차이를 오류로 출제했습니다. 문맥상 가능한 시제 변경 대신 수일치·능동/수동·관계사·분사·준동사 등 명백한 통사 오류 1개만 출제하세요."
                )
            elif re.search(r"(?:타당|적절|올바르|맞)(?:하다|다|습니다|음)(?!\s*(?:않|못))", exp_txt) and not re.search(
                r"(?:적절하지\s*않|올바르지\s*않|타당하지\s*않|맞지\s*않)", exp_txt
            ):
                issues.append(
                    "어법 문제 해설이 정답 밑줄을 '적절하다/타당하다'고 모순되게 설명했습니다. 명백한 문법 오류 1개만 출제하고 왜 틀렸는지 설명하세요."
                )
        elif "어휘" in t and ans in ("1", "2", "3", "4", "5"):
            # 2. 문맥상 어휘 적절성: 해설/치환목록에 적힌 교체 단어가 실제 지문 정답 밑줄에 반영되었는지 확인
            u_words = [re.sub(r"<[^>]+>", "", w).strip() for w in u_matches]
            ans_word = u_words[int(ans) - 1]
            ans_alpha = re.sub(r"[^a-zA-Z\-']", "", ans_word).lower()
            for entry in rw.replaced_terms_map or []:
                m_map = re.search(r"([A-Za-z][A-Za-z\-']*)\s*(?:->|→)\s*([A-Za-z][A-Za-z\-']*)", entry or "")
                if m_map:
                    orig_w, new_w = m_map.group(1).lower(), m_map.group(2).lower()
                    if orig_w != new_w and ans_alpha == orig_w:
                        issues.append(
                            f"문맥상 어휘 문제에서 정답 {ans}번 밑줄({ans_word!r})이 원문 그대로 남아 있고 "
                            f"교체 어휘({new_w!r})로 바뀌지 않았습니다. 지문 본문의 정답 밑줄을 실제로 교체하세요."
                        )
                        break
            if source_passage and ans_alpha:
                # 3. 문맥상 어휘 적절성: 5개 밑줄 단어 중 정확히 1개(정답 위치)만 원문과 다른 단어인지 검증
                not_in_src_indices: list[int] = []
                for idx_u, w in enumerate(u_words[:5]):
                    cleaned_w = re.sub(r"[^a-zA-Z\-]", "", w).lower()
                    if cleaned_w and not re.search(
                        r"\b" + re.escape(cleaned_w) + r"\b",
                        source_passage,
                        flags=re.IGNORECASE,
                    ):
                        not_in_src_indices.append(idx_u)
                if not not_in_src_indices:
                    issues.append(
                        f"문맥상 어휘 문제에서 5개 밑줄 단어({u_words[:5]})가 모두 원문 그대로입니다. "
                        f"반드시 정답 {ans}번 위치의 단어를 문맥상 부적절한 반의어로 지문 본문에서 직접 교체하세요."
                    )
                elif len(not_in_src_indices) == 1 and not_in_src_indices[0] != int(ans) - 1:
                    issues.append(
                        f"문맥상 어휘 문제의 정답 번호({ans}번)와 실제 지문에서 교체된 밑줄 위치({not_in_src_indices[0] + 1}번)가 일치하지 않습니다."
                    )
    elif "함축" in t:
        if u_count < 1:
            issues.append("함축적 의미 추론 지문에는 밑줄 친 표현 `<u>...</u>`가 최소 1개 포함되어야 합니다.")
    elif "빈칸" in t:
        if "____" not in stem:
            issues.append("빈칸 추론 지문에는 `___________` 빈칸 표기가 반드시 포함되어야 합니다.")
        else:
            # 2. 빈칸 포함 문장에서 빈칸을 제외했을 때 영문 동사가 하나도 남지 않는 불완전 문장 차단
            m_blk = re.search(r"_{4,}", stem)
            if m_blk:
                pre_txt = stem[: m_blk.start()]
                post_txt = stem[m_blk.end() :]
                s_start = max(pre_txt.rfind("\n"), pre_txt.rfind(". "), pre_txt.rfind("? "), pre_txt.rfind("! "))
                sent_left = pre_txt[s_start + 1 :] if s_start != -1 else pre_txt
                m_end = re.search(r"[.?!](?=\s|$|[\"'”’\)])|\n", post_txt)
                sent_right = post_txt[: m_end.start()] if m_end else post_txt
                rem_sentence = f"{sent_left} {sent_right}".strip()
                if not _sentence_has_english_verb(rem_sentence):
                    issues.append(
                        f"빈칸이 포함된 문장({rem_sentence!r})에서 빈칸을 제외하면 영문 동사가 전혀 남지 않습니다. "
                        "문장 전체(동사 포함)를 비우지 말고 주어·동사가 유지된 상태에서 핵심 목적어/보어/구만 빈칸으로 비우세요."
                    )

            # 3. 빈칸 앞뒤 구절/전치사 중복 차단
            if m_blk and ans in ("1", "2", "3", "4", "5") and len(rw.rewritten_choices or []) >= int(ans):
                before_toks = re.findall(r"[a-zA-Z\-']+", stem[: m_blk.start()].lower())
                after_toks = re.findall(r"[a-zA-Z\-']+", stem[m_blk.end() :].lower())
                ans_toks = re.findall(r"[a-zA-Z\-']+", (rw.rewritten_choices[int(ans) - 1] or "").lower())
                if len(ans_toks) >= 2:
                    if len(before_toks) >= 2 and before_toks[-2:] == ans_toks[:2]:
                        issues.append(
                            f"빈칸 직전 구절({' '.join(before_toks[-2:])})과 정답 선지의 시작 2단어가 중복됩니다. 중복을 제거해 다시 생성하세요."
                        )
                    elif len(after_toks) >= 2 and after_toks[:2] == ans_toks[-2:]:
                        issues.append(
                            f"빈칸 직후 구절({' '.join(after_toks[:2])})과 정답 선지의 끝 2단어가 중복됩니다. 중복을 제거해 다시 생성하세요."
                        )
                if after_toks and after_toks[0] in _REPEATABLE_PREPS and after_toks[0] in ans_toks[-4:]:
                    issues.append(
                        f"빈칸 바로 뒤 전치사({after_toks[0]!r})가 정답 선지 끝부분에도 중복 포함되어 비문이 됩니다. 전치사를 겹치지 않게 다시 생성하세요."
                    )
                elif before_toks and before_toks[-1] in (_REPEATABLE_PREPS | {"a", "an", "the"}) and ans_toks and ans_toks[0] == before_toks[-1]:
                    issues.append(
                        f"빈칸 바로 앞 단어({before_toks[-1]!r})가 정답 선지 시작 단어와 중복됩니다. 중복을 제거해 다시 생성하세요."
                    )
                elif (
                    before_toks
                    and before_toks[-1] in {"that", "because", "although", "though", "since", "when", "while", "if", "unless"}
                    and ans_toks
                    and ans_toks[0] in {"without", "with", "in", "on", "at", "by", "for", "from", "through"}
                ):
                    issues.append(
                        f"접속사({before_toks[-1]!r}) 바로 뒤 빈칸에 전치사구({ans_toks[0]!r} ...)만 들어가면 비문이 됩니다. 주어+동사를 포함한 절 또는 동사구로 출제하세요."
                    )
                post_raw = stem[m_blk.end() :]
                if re.match(r"^[ \t]+[A-Z]", post_raw) and not re.search(r"[.?!]$", pre_txt.rstrip()):
                    issues.append(
                        "문장 전체를 빈칸으로 비웠으나 빈칸 뒤에 마침표(.)가 누락되어 다음 문장과 붙었습니다. 문장 종결 부호를 유지하거나 문장 일부만 빈칸 처리하세요."
                    )
        if re.search(r"\(\s*[①②③④⑤]\s*\)|[①②③④⑤]", stem):
            issues.append(
                "빈칸 추론 지문에는 문장 삽입 번호 기호 `( ① )~( ⑤ )` 또는 `①~⑤`가 포함되면 안 됩니다. 모두 제거하세요."
            )
    elif "무관" in t:
        for c_num in ("①", "②", "③", "④", "⑤"):
            if c_num not in stem:
                issues.append(f"무관한 문장 본문에 번호 `{c_num}`가 누락되었습니다.")
                break
        if source_passage:
            src_cnt = _count_english_sentences(source_passage)
            gen_cnt = _count_english_sentences(stem)
            if src_cnt >= 3 and gen_cnt != src_cnt + 1:
                issues.append(
                    f"전체 흐름과 무관한 문장은 원문 문장({src_cnt}개)을 삭제하지 않고 무관한 새 문장 1개만 삽입해 "
                    f"총 {src_cnt + 1}문장이 되어야 합니다 (현재 {gen_cnt}문장). 원문을 보존하고 새 문장 1개를 삽입하세요."
                )
            elif _is_irrelevant_answer_from_source(stem, ans, source_passage):
                issues.append(
                    f"무관한 문장의 정답 {ans}번 문장이 원문 문장 그대로입니다. 원문 문장을 오답으로 우기지 말고 흐름과 무관한 새 문장을 삽입하세요."
                )
    elif "순서" in t:
        for mk in ("(A)", "(B)", "(C)"):
            if mk not in stem:
                issues.append(f"글의 순서 지문에 `{mk}` 단락 표기가 누락되었습니다.")
        if "[주어진 글]" not in stem and "주어진 글" not in stem:
            issues.append("글의 순서 지문에 `[주어진 글]` 섹션 헤더가 누락되었습니다.")
        if source_passage:
            # 4. 글의 순서 원문 문장 보존 검증 ([주어진 글] + (A) + (B) + (C) 합계 == 원문 문장 수)
            order_clean = re.sub(r"\[주어진\s*글\]", " ", stem)
            order_clean = re.sub(r"\(\s*[ABC]\s*\)", " ", order_clean)
            src_cnt = _count_english_sentences(source_passage)
            gen_cnt = _count_english_sentences(order_clean)
            if src_cnt >= 3 and gen_cnt != src_cnt:
                issues.append(
                    f"글의 순서 문제에서 원문 문장 수({src_cnt}개)와 [주어진 글]+(A)+(B)+(C) 합계 문장 수({gen_cnt}개)가 다릅니다. "
                    "원문 문장을 누락하거나 새 문장을 임의로 추가하지 말고 원문 그대로 배치하세요."
                )
            elif _is_order_abc_canonical(stem, source_passage):
                issues.append(
                    "글의 순서 문제에서 (A), (B), (C)가 원문 등장 순서 그대로((A)-(B)-(C) 정순) 배열되었습니다. "
                    "수능 5지선다에는 (A)-(B)-(C)가 없으므로 반드시 (B)-(A)-(C), (C)-(B)-(A) 등 단락 순서를 섞어 다시 출제하세요."
                )
            elif _has_order_scrambled_sentences(stem, source_passage):
                issues.append(
                    "글의 순서 문제에서 (A), (B), (C) 단락 내부에 원문의 비연속 문장이 뒤섞여 합쳐졌습니다. 각 단락은 원문에서 연속된 문장 묶음으로만 구성하세요."
                )
    elif "삽입" in t:
        if "[주어진 문장]" not in stem and "주어진 문장" not in stem:
            issues.append("문장 삽입 지문에 `[주어진 문장]` 섹션 헤더가 누락되었습니다.")
        for c_num in ("①", "②", "③", "④", "⑤"):
            if c_num not in stem:
                issues.append(f"문장 삽입 본문에 위치 기호 `{c_num}`가 누락되었습니다.")
                break
        # ( ① )~( ⑤ ) 마커가 문장 경계(. ? ! 뒤)가 아닌 주어·동사 한가운데 삽입되었는지 검증
        ins_body = re.split(r"\[주어진\s*문장\]", stem, maxsplit=1)[-1]
        ins_body_alpha = re.findall(r"[A-Za-z]+", ins_body)
        if len(ins_body_alpha) < 25:
            issues.append(
                f"문장 삽입 본문 영문 단어가 {len(ins_body_alpha)}개뿐입니다. [주어진 문장] 외에 전체 영어 지문 본문을 빠짐없이 포함하세요."
            )
        for m_ins in re.finditer(r"\(?\s*([①②③④⑤])\s*\)", ins_body):
            before = ins_body[: m_ins.start()].rstrip()
            if before and not re.search(r"[.?!][\"'”’\)]*$", before):
                issues.append(
                    f"문장 삽입 기호 `{m_ins.group(0).strip()}`가 문장 종결 부호(. ? !) 뒤의 문장 경계가 아닌 문장 중간에 배치되었습니다. 반드시 문장과 문장 사이 경계로 옮기세요."
                )
                break
    elif "요약" in t:
        if "[요약문]" not in stem and "요약문" not in stem:
            issues.append("요약문 완성 지문에 `[요약문]` 섹션 헤더가 누락되었습니다.")
        sum_section = re.split(r"\[요약문\]", stem, maxsplit=1)[-1]
        if not re.search(r"\(A\)\s*_{2,}", sum_section) or not re.search(r"\(B\)\s*_{2,}", sum_section):
            issues.append("요약문 섹션 안에 `(A) ________`와 `(B) ________` 빈칸 표기가 모두 포함되어야 합니다.")

    return issues


def _assemble_english_slim_stem(
    rw: RewrittenQuestion,
    canonical_type: str,
    clean_passage: str,
) -> None:
    """
    지문 전체 재출력 생략 유형(요지·주제·제목·요약문·문장 삽입·빈칸·함축)에서
    AI가 반환한 경량 결과(빈 문자열 / 요약 1문장 / 추출 1문장 / target_phrase)를
    원문(clean_passage)에 직접 결합·치환해 완성된 rewritten_stem을 조립한다.
    이미 전체 지문 형태로 출력된 경우(폴백 재시도 등)는 그대로 보존한다.
    """
    raw_stem = (rw.rewritten_stem or "").strip()
    t = canonical_type or ""

    if "요지" in t:
        if len(raw_stem) < max(60, int(len(clean_passage) * 0.55)):
            rw.rewritten_stem = f"다음 글의 요지로 가장 적절한 것은?\n{clean_passage}"
        return
    if "주제" in t:
        if len(raw_stem) < max(60, int(len(clean_passage) * 0.55)):
            rw.rewritten_stem = f"다음 글의 주제로 가장 적절한 것은?\n{clean_passage}"
        return
    if "제목" in t:
        if len(raw_stem) < max(60, int(len(clean_passage) * 0.55)):
            rw.rewritten_stem = f"다음 글의 제목으로 가장 적절한 것은?\n{clean_passage}"
        return

    if "요약" in t:
        # 이미 원문+요약문 전체가 들어있으면 그대로 두고, 요약문 1문장만 반환된 경우 원문과 결합
        if len(raw_stem) < max(120, int(len(clean_passage) * 0.65)):
            s_line = strip_english_embedded_choices(normalize_english_exam_text(raw_stem))
            s_line = re.sub(r"^[^\n]*가장\s*적절한\s*것은\??\s*", "", s_line).strip()
            s_line = re.sub(r"^\[?요약문\]?\s*", "", s_line).strip()
            if s_line:
                rw.rewritten_stem = (
                    "다음 글의 내용을 한 문장으로 요약하고자 한다. 빈칸 (A), (B)에 들어갈 말로 가장 적절한 것은?\n"
                    f"{clean_passage}\n\n"
                    "[요약문]\n"
                    f"{s_line}"
                )
        return

    if "삽입" in t:
        # 6문장 이상에서 추출 문장 1개만 반환된 경우 [주어진 문장] 헤더로 감싸 기계적 마커 배치로 넘김
        if (
            len(_split_english_sentences(clean_passage)) >= 6
            and "[주어진 문장]" not in raw_stem
            and len(raw_stem) < max(80, int(len(clean_passage) * 0.55))
        ):
            ext_s = re.sub(r"^[^\n]*가장\s*적절한\s*곳은\??\s*", "", raw_stem).strip()
            if ext_s:
                rw.rewritten_stem = f"[주어진 문장]\n{ext_s}"
        return

    if "빈칸" in t or "함축" in t:
        is_full_stem = len(raw_stem) >= max(80, int(len(clean_passage) * 0.6)) and (
            ("빈칸" in t and "____" in raw_stem) or ("함축" in t and "<u" in raw_stem.lower())
        )
        if is_full_stem or not raw_stem:
            return

        phrase = re.sub(r"</?u\b[^>]*>", "", raw_stem, flags=re.IGNORECASE).strip()
        phrase = re.sub(r"^[^\n]*가장\s*적절한\s*것은\??\s*", "", phrase).strip()
        phrase = phrase.strip("`\"'“”‘’")
        if not phrase:
            return

        exact_cnt = clean_passage.count(phrase)
        matched_span: tuple[int, int, str] | None = None
        if exact_cnt == 1:
            pos = clean_passage.index(phrase)
            matched_span = (pos, pos + len(phrase), phrase)
        elif exact_cnt == 0:
            ci_matches = list(re.finditer(re.escape(phrase), clean_passage, flags=re.IGNORECASE))
            if len(ci_matches) == 1:
                m_ci = ci_matches[0]
                matched_span = (m_ci.start(), m_ci.end(), m_ci.group(0))

        if matched_span is not None:
            st_i, en_i, orig_seg = matched_span
            if "빈칸" in t:
                replaced_body = f"{clean_passage[:st_i]}___________{clean_passage[en_i:]}"
                rw.rewritten_stem = f"다음 빈칸에 들어갈 말로 가장 적절한 것은?\n{replaced_body}"
            else:
                replaced_body = f"{clean_passage[:st_i]}<u>{orig_seg}</u>{clean_passage[en_i:]}"
                rw.rewritten_stem = f"밑줄 친 부분이 다음 글에서 의미하는 바로 가장 적절한 것은?\n{replaced_body}"
        return

    if "순서" in t:
        # 이미 완성된 (A)(B)(C) 전체 지문이 반환된 경우(폴백 재시도 등)는 그대로 보존
        if "(A)" in raw_stem and "(B)" in raw_stem and "(C)" in raw_stem and len(raw_stem) >= max(100, int(len(clean_passage) * 0.6)):
            return
        m_sp = re.search(r"SPLIT\s*:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", raw_stem, flags=re.IGNORECASE)
        sents = _split_english_sentences(clean_passage)
        if m_sp and len(sents) >= 4:
            s1, s2, s3 = int(m_sp.group(1)), int(m_sp.group(2)), int(m_sp.group(3))
            if 0 < s1 < s2 < s3 < len(sents):
                intro = " ".join(sents[:s1]).strip()
                orig_blocks = [
                    " ".join(sents[s1:s2]).strip(),
                    " ".join(sents[s2:s3]).strip(),
                    " ".join(sents[s3:]).strip(),
                ]
                if all(orig_blocks):
                    # 5지선다 오순 조합: (A)-(C)-(B)=1, (B)-(A)-(C)=2, (B)-(C)-(A)=3, (C)-(A)-(B)=4, (C)-(B)-(A)=5
                    # 각 조합에서 (A), (B), (C)에 어떤 원문 덩어리(0,1,2)를 넣으면 정답이 해당 순서가 되는지 매핑
                    order_specs = [
                        ("1", "(A) - (C) - (B)", (0, 2, 1)),  # A=0, B=2, C=1 -> 원문순서 0,1,2 = (A)-(C)-(B)
                        ("2", "(B) - (A) - (C)", (1, 0, 2)),  # A=1, B=0, C=2 -> 원문순서 0,1,2 = (B)-(A)-(C)
                        ("3", "(B) - (C) - (A)", (2, 0, 1)),  # A=2, B=0, C=1 -> 원문순서 0,1,2 = (B)-(C)-(A)
                        ("4", "(C) - (A) - (B)", (1, 2, 0)),  # A=1, B=2, C=0 -> 원문순서 0,1,2 = (C)-(A)-(B)
                        ("5", "(C) - (B) - (A)", (2, 1, 0)),  # A=2, B=1, C=0 -> 원문순서 0,1,2 = (C)-(B)-(A)
                    ]
                    ans_num, ans_seq_str, perm = random.choice(order_specs)
                    block_a = orig_blocks[perm[0]]
                    block_b = orig_blocks[perm[1]]
                    block_c = orig_blocks[perm[2]]
                    rw.rewritten_stem = (
                        "주어진 글 다음에 이어질 글의 순서로 가장 적절한 것은?\n"
                        "[주어진 글]\n"
                        f"{intro}\n\n"
                        f"(A) {block_a}\n\n"
                        f"(B) {block_b}\n\n"
                        f"(C) {block_c}"
                    )
                    rw.rewritten_choices = [
                        "(A) - (C) - (B)",
                        "(B) - (A) - (C)",
                        "(B) - (C) - (A)",
                        "(C) - (A) - (B)",
                        "(C) - (B) - (A)",
                    ]
                    rw.preserved_answer = ans_num
                    if rw.pdf_grounded_explanation:
                        rw.pdf_grounded_explanation = _ORDER_SEQ_RE.sub(ans_seq_str, rw.pdf_grounded_explanation)
                    else:
                        rw.pdf_grounded_explanation = f"주어진 글 뒤에 논리적 흐름에 따라 {ans_seq_str} 순서로 이어지는 것이 자연스럽습니다."
        return

    if "무관" in t:
        if all(mk in raw_stem for mk in ("①", "②", "③", "④", "⑤")) and len(raw_stem) >= max(100, int(len(clean_passage) * 0.6)):
            return
        m_pos = re.search(r"INSERT_AT\s*:\s*([1-5])", raw_stem, flags=re.IGNORECASE)
        m_sent = re.search(r"SENTENCE\s*:\s*(.+)", raw_stem, flags=re.IGNORECASE | re.DOTALL)
        sents = _split_english_sentences(clean_passage)
        if m_sent and len(sents) >= 5:
            ins_sent = re.sub(r"\s+", " ", m_sent.group(1)).strip().strip("\"'")
            if ins_sent and not re.search(r"[.?!]$", ins_sent):
                ins_sent += "."
            if len(re.findall(r"[A-Za-z]+", ins_sent)) >= 5:
                insert_pos = int(m_pos.group(1)) if m_pos else random.choice([2, 3, 4])
                insert_pos = max(2, min(4, insert_pos))
                intro_cnt = 2 if len(sents) >= 6 else 1
                intro_part = " ".join(sents[:intro_cnt]).strip()
                body_orig = sents[intro_cnt:]
                numbered_orig = body_orig[:4]
                tail_orig = body_orig[4:]
                merged_five: list[str] = []
                orig_iter = iter(numbered_orig)
                for slot in range(1, 6):
                    if slot == insert_pos:
                        merged_five.append(ins_sent)
                    else:
                        try:
                            merged_five.append(next(orig_iter))
                        except StopIteration:
                            break
                if len(merged_five) == 5:
                    syms = ["①", "②", "③", "④", "⑤"]
                    marked_str = " ".join(f"{syms[i]} {merged_five[i]}" for i in range(5))
                    tail_str = (" " + " ".join(tail_orig)) if tail_orig else ""
                    rw.rewritten_stem = (
                        "다음 글에서 전체 흐름과 관계없는 문장은?\n"
                        f"{intro_part} {marked_str}{tail_str}".strip()
                    )
                    rw.rewritten_choices = list(syms)
                    rw.preserved_answer = str(insert_pos)
                    c_sym = syms[insert_pos - 1]
                    if rw.pdf_grounded_explanation:
                        rw.pdf_grounded_explanation = re.sub(
                            r"(?:[1-5]\s*번|[①②③④⑤](?:\s*번)?)",
                            f"{c_sym}번",
                            rw.pdf_grounded_explanation,
                            count=1,
                        )
        return

    if ("어법" in t or ("어휘" in t and "선택형" not in t and "(A)" not in t)) and not (
        "<u" in raw_stem.lower() and len(raw_stem) >= max(100, int(len(clean_passage) * 0.6))
    ):
        # 5줄 슬림 포맷 파싱: "1: orig" 또는 "3: orig -> replaced"
        spec_lines: dict[int, tuple[str, str]] = {}
        for ln in raw_stem.splitlines():
            m_ln = re.match(r"^\s*(?:([1-5])|([①②③④⑤]))\s*[:.)]\s*(.+?)\s*$", ln)
            if not m_ln:
                continue
            num_str = m_ln.group(1) or _CIRCLED_DIGIT_TO_NUM.get(m_ln.group(2) or "", "")
            if not num_str:
                continue
            idx_n = int(num_str)
            body_spec = m_ln.group(3).strip()
            if "->" in body_spec or "→" in body_spec:
                parts = re.split(r"\s*(?:->|→)\s*", body_spec, maxsplit=1)
                orig_p = parts[0].strip().strip("`\"'“”‘’")
                repl_p = parts[1].strip().strip("`\"'“”‘’")
            else:
                orig_p = body_spec.strip("`\"'“”‘’")
                repl_p = orig_p
            if orig_p:
                spec_lines[idx_n] = (orig_p, repl_p)

        if len(spec_lines) == 5:
            syms = ["①", "②", "③", "④", "⑤"]
            cursor = 0
            out_parts: list[str] = []
            display_words: list[str] = []
            changed_indices: list[int] = []
            matched_all = True
            for i in range(1, 6):
                orig_p, repl_p = spec_lines[i]
                # 커서 이후에서 단어 경계 우선 매칭, 실패 시 단순 부분 문자열 매칭
                seg_after = clean_passage[cursor:]
                m_word = re.search(rf"(?<![A-Za-z]){re.escape(orig_p)}(?![A-Za-z])", seg_after)
                if m_word is None:
                    m_word = re.search(re.escape(orig_p), seg_after, flags=re.IGNORECASE)
                if m_word is None:
                    matched_all = False
                    break
                abs_st = cursor + m_word.start()
                abs_en = cursor + m_word.end()
                out_parts.append(clean_passage[cursor:abs_st])
                shown = repl_p if repl_p != orig_p else clean_passage[abs_st:abs_en]
                out_parts.append(f"{syms[i - 1]} <u>{shown}</u>")
                display_words.append(shown)
                if repl_p.lower() != orig_p.lower():
                    changed_indices.append(i)
                cursor = abs_en

            if matched_all and len(changed_indices) == 1:
                out_parts.append(clean_passage[cursor:])
                assembled_body = "".join(out_parts)
                instr = (
                    "다음 글의 밑줄 친 부분 중, 어법상 틀린 것은?"
                    if "어법" in t
                    else "다음 글의 밑줄 친 부분 중, 문맥상 낱말의 쓰임이 적절하지 않은 것은?"
                )
                rw.rewritten_stem = f"{instr}\n{assembled_body}"
                rw.preserved_answer = str(changed_indices[0])
                if "어법" in t:
                    rw.rewritten_choices = list(syms)
                else:
                    rw.rewritten_choices = display_words


_INCOMPLETE_BLANK_RE = re.compile(
    r"_{2,}|"
    r"\[\s+\]|\(\s+\)|"
    r"\(\s*[A-C]\s*\)\s*(?:\[[^\]]+/[^\]]+\]|[A-Za-z-]+/[A-Za-z-]+)|"
    r"[A-Za-z][ \t]{2,}[.?!](?=\s|$|[\"'”’\)])|"
    r",\s*[.?!](?=\s|$|[\"'”’\)])|"
    r"\b(?:isn't\s+much|aren't\s+many|has\s+to|have\s+to|had\s+to|want\s+to|need\s+to|"
    r"able\s+to|going\s+to|ought\s+to|used\s+to|"
    r"that|which|who|whom|whose|whether|because|although|though|while|unless|until|whereas|"
    r"of|to|in|on|at|for|with|by|from|into|than|between|through|without|about|"
    r"the|a|an|this|these|those|their|our|its|my|his|her)\s*[.?!]"
    r"(?=\s|$|[\"'”’\)])",
    re.IGNORECASE,
)

_DANGLING_VERB_END_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|can|could|will|would|shall|should|may|might|must|do|does|did|have|has|had)\s*[.?!](?=\s|$|[\"'”’\)])",
    re.IGNORECASE,
)

_EXAM_STRUCTURE_MARKER_RE = re.compile(
    r"[①②③④⑤]|"
    r"</?u\b[^>]*>|"
    r"\[(?:주어진\s*글|주어진\s*문장)\]|"
    r"(?:^|\n|(?<=[.?!\"'”’\)]\s))\(\s*[ABC]\s*\)\s+(?=[A-Z\"'“‘\(])",
    re.IGNORECASE,
)

_RESTORE_TARGET_TYPE_KEYWORDS = (
    "어법",
    "어휘",
    "빈칸",
    "무관",
    "흐름",
    "삽입",
    "순서",
    "낱말",
)


def _has_exam_structure_markers(passage: str) -> bool:
    """영어 지문에 미정리 기출 마커(①~⑤, (A)(B)(C), <u>, [주어진 글/문장])가 남아 있는지 판별한다."""
    if not passage:
        return False
    return bool(_EXAM_STRUCTURE_MARKER_RE.search(passage))


_NON_VERB_S_WORDS: frozenset[str] = frozenset(
    {
        "this", "these", "those", "his", "hers", "its", "ours", "theirs", "yours",
        "thus", "less", "plus", "across", "whereas", "perhaps", "always", "sometimes",
        "unless", "regardless", "nevertheless", "process", "progress", "success",
        "access", "business", "analysis", "emphasis", "basis", "crisis", "species", "series",
        "towards", "besides", "sometimes", "always", "often", "several", "various",
    }
)


def _has_incomplete_english_blank(passage: str) -> bool:
    """원문 영어 지문 자체가 빈칸·선택형 기출이어서 비문/누락 문장이 있는지 판별한다."""
    if not passage:
        return False
    if _INCOMPLETE_BLANK_RE.search(passage):
        return True
    for sent in _split_english_sentences(passage):
        words = re.findall(r"[A-Za-z]+", sent)
        low_words = [w.lower() for w in words]
        if len(words) >= 6 and _DANGLING_VERB_END_RE.search(sent):
            # 'So it is.' 같은 짧은 도치·응답문이 아닌 긴 문장이 be/조동사로 끝나면 빈칸 절단 문장
            if len(low_words) < 2 or low_words[-2] not in {"it", "they", "we", "he", "she", "there", "so", "nor"}:
                return True
        if len(words) >= 5 and not _sentence_has_english_verb(sent):
            # 일반 3인칭 단수 동사(-s/-es)까지 오탐하지 않도록 추가 확인 (단, 관사·지시사 뒤 '-ing' 명사 및 비동사 '-s' 제외)
            has_likely_verb = False
            for idx_w, w in enumerate(low_words):
                if w in _NON_VERB_S_WORDS or w in _DETERMINER_SET:
                    continue
                if len(w) >= 4 and w.endswith(("s", "ed")):
                    has_likely_verb = True
                    break
                if len(w) >= 5 and w.endswith("ing"):
                    if idx_w > 0 and low_words[idx_w - 1] in _DETERMINER_SET:
                        continue
                    has_likely_verb = True
                    break
            if not has_likely_verb:
                return True
    return False


def _needs_english_passage_restoration(passage: str, orig_question_type: str = "") -> bool:
    """
    기출 지문에 실제 미복원 흔적(빈칸 `____`, `①~⑤`, `(A)(B)(C)`, `<u>`, 불완전 문장)이 있을 때만
    원문 복원/교정이 필요한 것으로 판별한다.
    이미 탭 1에서 복원이 완료된 깨끗한 지문은 재복원(원문 변조·문장 삭제 위험)하지 않는다.
    """
    _ = orig_question_type
    if not passage or not passage.strip():
        return False
    if _has_incomplete_english_blank(passage):
        return True
    if _has_exam_structure_markers(passage):
        return True
    return False

