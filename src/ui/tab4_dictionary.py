from __future__ import annotations

import html as _html
import json
import random
import re
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from src.auth import resolve_active_gemini_api_key
from src.config import FLASH_LITE_MODEL
from src.gemini.dictionary import lookup_english_word_with_gemini
from src.models import (
    EnglishDictEntry,
    QuestionRecord,
    UserVocabItem,
    normalize_english_dict_key,
)
from src.ui.common import ensure_token_quota, sanitize_id_part, usage_scope
from src.ui.context import AppContext
from src.ui_styles import render_chips
from src.vocab_pdf_generator import generate_vocab_wordbook_pdf

# 객관식 퀴즈 시 보기가 4개 미만일 때 오답 선지를 보충하기 위한 수능 필수 어휘 기본 풀
_FALLBACK_DISTRACTOR_BANK: tuple[tuple[str, str], ...] = (
    ("ubiquitous", "어디에나 존재하는, 도처에 있는"),
    ("ephemeral", "일시적인, 수명이 짧은"),
    ("pragmatic", "실용적인, 현실적인"),
    ("ambiguous", "모호한, 여러 가지로 해석될 수 있는"),
    ("resilience", "회복탄력성, (충격에서의) 빠른 회복"),
    ("coherent", "일관성 있는, 논리 정연한"),
    ("plausible", "그럴듯한, 타당해 보이는"),
    ("arbitrary", "임의적인, 제멋대로인"),
    ("inevitable", "피할 수 없는, 필연적인"),
    ("paradox", "역설, 모순"),
    ("subtle", "미묘한, 포착하기 어려운"),
    ("implicit", "암묵적인, 내포된"),
    ("explicit", "명시적인, 분명한"),
    ("adjacent", "인접한, 가까운"),
    ("authentic", "진정한, 진짜의"),
    ("cognitive", "인지의, 인식에 관한"),
    ("deteriorate", "악화되다, 저하되다"),
    ("alleviate", "완화하다, 덜어주다"),
    ("suppress", "억압하다, 참다, 숨기다"),
    ("compel", "강요하다, ~하게 만들다"),
    ("attribute", "~의 탓(덕)으로 돌리다, 속성"),
    ("consolidate", "통합하다, 공고히 하다"),
    ("discrepancy", "불일치, 차이"),
    ("scrutinize", "면밀히 조사하다"),
)


def _parse_key_vocab_entry(raw: str) -> tuple[str, str]:
    """'resilience: 회복탄력성' · 'resilience (회복탄력성)' · 'take into account - 고려하다' 등을 (표제어, 뜻)으로 분리한다."""
    txt = (raw or "").strip()
    if not txt:
        return "", ""

    for sep in (":", "：", "—", "–", " - "):
        if sep in txt:
            left, right = txt.split(sep, 1)
            w = left.strip()
            m = right.strip()
            if w:
                return w, m

    m_paren = re.match(r"^([A-Za-z][A-Za-z0-9'\-\s]{0,60}?)\s*\(([^()]+)\)\s*$", txt)
    if m_paren:
        return m_paren.group(1).strip(), m_paren.group(2).strip()

    return txt, ""


def _build_word_match_pattern(term: str) -> re.Pattern[str] | None:
    """단어·숙어 및 흔한 굴절형(-s/-es/-ies, -d/-ed/-ied, -ing)까지 탐지하는 정규식을 생성한다."""
    clean = normalize_english_dict_key(term)
    if not clean or not re.search(r"[a-z]", clean):
        return None
    words = clean.split()
    if len(words) == 1:
        base = re.escape(words[0])
        if len(words[0]) >= 3 and words[0].endswith("e"):
            stem = re.escape(words[0][:-1])
            pat = rf"\b(?:{base}(?:s|d|r)?|{stem}(?:ed|ing|es|er|est))\b"
        elif len(words[0]) >= 4 and words[0].endswith("y") and words[0][-2] not in "aeiou":
            stem = re.escape(words[0][:-1])
            pat = rf"\b(?:{base}(?:s|d)?|{stem}(?:ies|ied|ier|iest|ying)|{base}ing)\b"
        else:
            pat = rf"\b{base}(?:s|es|d|ed|ing|er|est)?\b"
        return re.compile(pat, re.IGNORECASE)

    joined = r"\s+".join(re.escape(w) for w in words)
    return re.compile(rf"\b{joined}(?:s|es|ed|ing)?\b", re.IGNORECASE)


def _highlight_sentence_html(sentence: str, patterns: list[re.Pattern[str]]) -> str:
    """영문 문장에서 매칭 구간만 안전하게 HTML <mark>로 강조한다 (XSS 안전)."""
    spans: list[tuple[int, int]] = []
    for pat in patterns:
        for m in pat.finditer(sentence):
            spans.append((m.start(), m.end()))
    if not spans:
        return _html.escape(sentence)

    spans.sort()
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    out: list[str] = []
    pos = 0
    for s, e in merged:
        if pos < s:
            out.append(_html.escape(sentence[pos:s]))
        out.append(
            "<mark style='background:#FEF08A; color:#854D0E; padding:0 3px; border-radius:3px; font-weight:700;'>"
            + _html.escape(sentence[s:e])
            + "</mark>"
        )
        pos = e
    if pos < len(sentence):
        out.append(_html.escape(sentence[pos:]))
    return "".join(out)


def find_word_in_exam_passages(
    records: list[QuestionRecord],
    headword: str,
    raw_query: str = "",
    max_hits: int = 8,
) -> list[dict[str, Any]]:
    """저장된 영어 기출 DB 지문과 핵심 어휘에서 해당 단어가 실제 등장한 문장과 출처를 검색한다."""
    patterns: list[re.Pattern[str]] = []
    seen_terms: set[str] = set()
    for term in (headword, raw_query):
        norm = normalize_english_dict_key(term)
        if not norm or norm in seen_terms:
            continue
        seen_terms.add(norm)
        pat = _build_word_match_pattern(norm)
        if pat is not None:
            patterns.append(pat)

    if not patterns:
        return []

    hits: list[dict[str, Any]] = []
    seen_passages: set[str] = set()

    for rec in records:
        if getattr(rec, "subject_category", "") != "영어":
            continue
        passage = (getattr(rec, "passage_text", "") or "").strip()
        q_text = (getattr(rec, "question_text", "") or "").strip()
        target_text = passage or q_text
        kv_list = list(getattr(rec, "key_vocabulary", []) or [])
        kv_matches = [
            kv
            for kv in kv_list
            if any(pat.search(kv) for pat in patterns)
        ]
        matched_in_text = bool(target_text and any(pat.search(target_text) for pat in patterns))
        if not matched_in_text and not kv_matches:
            continue

        # 장문 41~42 / 43~45 공유 지문 중복 방지
        grp_id = (getattr(rec, "passage_group_id", "") or "").strip()
        dedup_key = f"{rec.exam_name}||{grp_id}" if grp_id else f"{rec.exam_name}||{rec.q_num}"
        if dedup_key in seen_passages:
            continue
        seen_passages.add(dedup_key)

        matched_sentence = ""
        if target_text:
            flat = re.sub(r"\s+", " ", target_text).strip()
            sentences = re.split(r"(?<=[.?!])\s+", flat)
            for sent in sentences:
                if any(pat.search(sent) for pat in patterns):
                    matched_sentence = sent[:420] + ("..." if len(sent) > 420 else "")
                    break
            if not matched_sentence and matched_in_text:
                matched_sentence = flat[:320] + ("..." if len(flat) > 320 else "")

        hits.append(
            {
                "exam_name": rec.exam_name,
                "q_num": rec.q_num,
                "group_id": grp_id,
                "question_type": getattr(rec, "english_question_type", "") or rec.unit_name or "독해",
                "passage_topic": getattr(rec, "passage_topic", "") or "",
                "sentence": matched_sentence,
                "sentence_html": _highlight_sentence_html(matched_sentence, patterns) if matched_sentence else "",
                "key_vocab_matches": kv_matches,
            }
        )
        if len(hits) >= max_hits:
            break

    return hits


def _entry_to_user_vocab_item(entry: EnglishDictEntry, source_exam: str = "") -> UserVocabItem:
    pos_list: list[str] = []
    for s in entry.senses:
        p = (s.pos or "").strip()
        if p and p not in pos_list:
            pos_list.append(p)
    first_ex_en = next((s.example_en for s in entry.senses if (s.example_en or "").strip()), "")
    first_ex_ko = next((s.example_ko for s in entry.senses if (s.example_en or "").strip()), "")
    return UserVocabItem(
        headword=entry.headword.strip(),
        normalized_key=normalize_english_dict_key(entry.headword),
        summary_ko=entry.summary_ko.strip(),
        phonetic=entry.phonetic.strip(),
        cefr_level=entry.cefr_level.strip(),
        pos_summary=" · ".join(pos_list[:3]),
        synonyms=list(entry.synonyms[:5]),
        example_en=first_ex_en,
        example_ko=first_ex_ko,
        source_exam=source_exam.strip(),
    )


def _render_tts_buttons(word: str) -> None:
    """브라우저 내장 Web Speech API를 이용해 미국식 발음 듣기 버튼을 렌더링한다."""
    clean_word = (word or "").strip()
    if not clean_word or not re.search(r"[A-Za-z]", clean_word):
        return
    js_word = json.dumps(clean_word)
    widget_html = f"""
    <div style="display:flex; gap:6px; align-items:center; flex-wrap:wrap; font-family:sans-serif;">
      <button type="button" onclick='(function(){{
        if(!("speechSynthesis" in window)) return;
        window.speechSynthesis.cancel();
        var u = new SpeechSynthesisUtterance({js_word});
        u.lang = "en-US";
        u.rate = 0.96;
        window.speechSynthesis.speak(u);
      }})()' style="cursor:pointer; background:#EFF6FF; color:#1D4ED8; border:1px solid #93C5FD;
        border-radius:999px; padding:4px 11px; font-size:12.5px; font-weight:700;">
        🔊 미국식 발음 듣기
      </button>
      <button type="button" onclick='(function(){{
        if(!("speechSynthesis" in window)) return;
        window.speechSynthesis.cancel();
        var u = new SpeechSynthesisUtterance({js_word});
        u.lang = "en-US";
        u.rate = 0.70;
        window.speechSynthesis.speak(u);
      }})()' style="cursor:pointer; background:#F8FAFC; color:#334155; border:1px solid #CBD5E1;
        border-radius:999px; padding:4px 10px; font-size:12px; font-weight:600;">
        🐢 느리게
      </button>
    </div>
    """
    if hasattr(st, "iframe"):
        st.iframe(widget_html, height=40)
    else:
        components.html(widget_html, height=40)


def _execute_dict_lookup(
    ctx: AppContext,
    raw_query: str,
    *,
    force_refresh: bool = False,
) -> tuple[EnglishDictEntry | None, str, str]:
    """공용 캐시 확인 -> 미스 시 기출 문맥 포함 Gemini 조회 및 캐시 저장."""
    q = (raw_query or "").strip()
    if not q:
        return None, "", "검색할 영단어나 숙어를 입력해 주세요."

    if not force_refresh:
        cached = ctx.storage.get_cached_english_dict_entry(q)
        if cached is not None:
            return cached, "cache", ""

    api_key, key_label = resolve_active_gemini_api_key(
        ctx.current_user,
        ctx.creds.gemini_api_key,
        model_name=FLASH_LITE_MODEL,
    )
    if not api_key:
        return None, "", f"영단어 AI 사전 조회용 Gemini API 키({key_label})가 설정되지 않았습니다."
    if not ensure_token_quota(ctx):
        return None, "", ""

    records = ctx.storage.load_db()
    ctx_hits = find_word_in_exam_passages(records, q, raw_query=q, max_hits=2)
    ctx_sentences = [h["sentence"] for h in ctx_hits if h.get("sentence")]

    try:
        with usage_scope(ctx):
            entry = lookup_english_word_with_gemini(
                api_key=api_key,
                model_name=FLASH_LITE_MODEL,
                query=q,
                context_sentences=ctx_sentences,
            )
        ctx.storage.save_cached_english_dict_entry(entry, aliases=[q])
        return entry, "ai", ""
    except Exception as exc:
        return None, "", f"영단어 조회 중 오류가 발생했습니다: {exc}"


def _render_dict_entry_card(ctx: AppContext, entry: EnglishDictEntry, origin_tag: str, raw_query: str) -> None:
    """조회된 영단어 사전 엔트리 카드, 발음 버튼, 단어장 저장, 기출 지문 용례를 렌더링한다."""
    username = ctx.current_user.username if ctx.current_user else ""
    norm_key = normalize_english_dict_key(entry.headword)
    user_vocab = ctx.storage.load_user_vocab(username) if username else []
    saved_item = next((v for v in user_vocab if v.normalized_key == norm_key), None)

    records = ctx.storage.load_db()
    exam_hits = find_word_in_exam_passages(records, entry.headword, raw_query=raw_query, max_hits=8)

    with st.container(border=True):
        top_c1, top_c2 = st.columns([3.0, 1.35])
        with top_c1:
            lvl_badge = (
                f"<span style='background:#DCFCE7; color:#166534; font-size:0.78rem; font-weight:700; "
                f"padding:2px 8px; border-radius:999px; margin-left:6px;'>{_html.escape(entry.cefr_level)}</span>"
                if entry.cefr_level
                else ""
            )
            src_badge = (
                "<span style='background:#EFF6FF; color:#1D4ED8; font-size:0.75rem; font-weight:700; "
                "padding:2px 8px; border-radius:999px; margin-left:6px;'>⚡ 공용 사전 DB (토큰 0)</span>"
                if origin_tag == "cache"
                else "<span style='background:#FEF3C7; color:#92400E; font-size:0.75rem; font-weight:700; "
                "padding:2px 8px; border-radius:999px; margin-left:6px;'>✨ AI 정밀 생성 · DB 저장됨</span>"
            )
            ipa_span = (
                f"<span style='color:#64748B; font-size:1.02rem; margin-left:8px;'>{_html.escape(entry.phonetic)}</span>"
                if entry.phonetic
                else ""
            )
            st.markdown(
                f"<div style='display:flex; align-items:baseline; flex-wrap:wrap; gap:4px;'>"
                f"<span style='font-size:1.72rem; font-weight:800; color:#0F172A;'>{_html.escape(entry.headword)}</span>"
                f"{ipa_span}{lvl_badge}{src_badge}"
                f"</div>",
                unsafe_allow_html=True,
            )
        with top_c2:
            if saved_item is not None:
                st.success("✅ 내 단어장 저장됨")
                if st.button("🔄 내 단어장 정보 최신화", key=f"tab4_resave_{norm_key}", width="stretch"):
                    src_ex = f"{exam_hits[0]['exam_name']} {exam_hits[0]['q_num']}번" if exam_hits else ""
                    ctx.storage.save_user_vocab_item(username, _entry_to_user_vocab_item(entry, source_exam=src_ex))
                    st.toast(f"'{entry.headword}' 단어장 정보를 업데이트했습니다.")
                    st.rerun()
            else:
                if st.button("⭐ 내 단어장에 저장", type="primary", key=f"tab4_save_{norm_key}", width="stretch"):
                    src_ex = f"{exam_hits[0]['exam_name']} {exam_hits[0]['q_num']}번" if exam_hits else ""
                    ctx.storage.save_user_vocab_item(username, _entry_to_user_vocab_item(entry, source_exam=src_ex))
                    st.toast(f"'{entry.headword}'을(를) 내 단어장에 저장했습니다.")
                    st.rerun()

        _render_tts_buttons(entry.headword)

        if entry.summary_ko:
            st.markdown(
                f"<div style='background:#F8FAFC; border-left:4px solid #2563EB; padding:0.6rem 0.9rem; "
                f"border-radius:0 8px 8px 0; margin:0.45rem 0 0.8rem 0; font-size:1.05rem; font-weight:700; color:#1E293B;'>"
                f"{_html.escape(entry.summary_ko)}</div>",
                unsafe_allow_html=True,
            )

        if entry.senses:
            st.markdown("##### 📖 품사별 핵심 의미 & 수능 예문")
            for idx, sense in enumerate(entry.senses, start=1):
                nuance_html = (
                    f"<div style='font-size:0.84rem; color:#475569; margin-top:3px;'>💡 <b>뉘앙스:</b> {_html.escape(sense.nuance_note)}</div>"
                    if sense.nuance_note
                    else ""
                )
                ex_html = ""
                if sense.example_en:
                    ex_html = (
                        f"<div style='background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:0.45rem 0.7rem; margin-top:6px;'>"
                        f"<div style='font-size:0.9rem; color:#0F172A;'>{_html.escape(sense.example_en)}</div>"
                        f"<div style='font-size:0.83rem; color:#64748B; margin-top:2px;'>{_html.escape(sense.example_ko)}</div>"
                        f"</div>"
                    )
                st.markdown(
                    f"<div style='margin-bottom:0.65rem; padding-bottom:0.55rem; border-bottom:1px solid #F1F5F9;'>"
                    f"<div><span style='background:#DBEAFE; color:#1E40AF; font-size:0.78rem; font-weight:700; "
                    f"padding:1px 7px; border-radius:4px;'>{_html.escape(sense.pos)}</span> "
                    f"<b style='font-size:0.98rem; color:#0F172A;'>{idx}. {_html.escape(sense.meaning_ko)}</b></div>"
                    f"{nuance_html}{ex_html}</div>",
                    unsafe_allow_html=True,
                )

        # 유의어 / 반의어 / 파생어 / 연어
        rel_c1, rel_c2 = st.columns(2)
        with rel_c1:
            if entry.synonyms:
                st.markdown("**🔁 수능 패러프레이징 유의어**")
                render_chips(entry.synonyms, variant="blue", limit=10)
            if entry.derivatives:
                st.markdown("**🧩 파생어**")
                render_chips(entry.derivatives, variant="green", limit=10)
        with rel_c2:
            if entry.antonyms:
                st.markdown("**⚖️ 반의어**")
                render_chips(entry.antonyms, variant="red", limit=10)
            if entry.collocations:
                st.markdown("**🔗 빈출 연어 · 숙어 결합**")
                render_chips(entry.collocations, variant="amber", limit=10)

        # 유의어 꼬리물기 바로 검색 버튼 (최대 4개)
        syn_words: list[str] = []
        for s_item in entry.synonyms[:4]:
            w_only, _ = _parse_key_vocab_entry(s_item)
            w_only = re.sub(r"\s*\([^)]*\)\s*", "", w_only).strip()
            if w_only and re.fullmatch(r"[A-Za-z][A-Za-z'\-\s]{0,40}", w_only):
                syn_words.append(w_only)
        if syn_words:
            q_cols = st.columns(min(4, len(syn_words)) + 2)
            for si, sw in enumerate(syn_words):
                with q_cols[si]:
                    if st.button(f"🔎 {sw}", key=f"tab4_syn_jump_{norm_key}_{si}", width="stretch"):
                        st.session_state["tab4_dict_pending_query"] = sw
                        st.rerun()

        if entry.etymology_tip or entry.csat_tip:
            tip_c1, tip_c2 = st.columns(2)
            with tip_c1:
                if entry.etymology_tip:
                    st.info(f"**🌱 어원 · 암기 팁**\n\n{entry.etymology_tip}")
            with tip_c2:
                if entry.csat_tip:
                    st.warning(f"**🎯 수능 · 평가원 출제 포인트**\n\n{entry.csat_tip}")

    # 저장된 영어 기출 지문 내 실제 등장 용례 연동
    with st.container(border=True):
        st.markdown(f"##### 📝 저장된 영어 기출 지문 속 `{entry.headword}` 실제 용례 ({len(exam_hits)}건)")
        if not exam_hits:
            st.caption(
                "현재 DB에 저장된 영어 모의고사 지문에서는 일치하는 문장이 아직 발견되지 않았습니다. "
                "([탭 1]에서 영어 모의고사를 추가하면 자동으로 지문 속 문맥이 연동됩니다.)"
            )
        else:
            for h in exam_hits:
                grp_badge = f" [{h['group_id']}번 세트]" if h.get("group_id") else ""
                topic_txt = f" · 주제: {h['passage_topic']}" if h.get("passage_topic") else ""
                kv_txt = (
                    f"<div style='font-size:0.78rem; color:#1D4ED8; margin-top:3px;'>"
                    f"핵심 어휘 태그: {_html.escape(', '.join(h['key_vocab_matches']))}</div>"
                    if h.get("key_vocab_matches")
                    else ""
                )
                sent_box = (
                    f"<div style='font-size:0.9rem; line-height:1.55; color:#1E293B; margin-top:4px;'>{h['sentence_html']}</div>"
                    if h.get("sentence_html")
                    else ""
                )
                st.markdown(
                    f"<div style='background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:0.55rem 0.75rem; margin-bottom:0.45rem;'>"
                    f"<div style='font-size:0.8rem; color:#475569; font-weight:700;'>"
                    f"📄 {_html.escape(str(h['exam_name']))} · {h['q_num']}번{_html.escape(grp_badge)} "
                    f"({ _html.escape(str(h['question_type'])) }){_html.escape(topic_txt)}</div>"
                    f"{sent_box}{kv_txt}</div>",
                    unsafe_allow_html=True,
                )


def _render_dict_search_subtab(ctx: AppContext) -> None:
    st.subheader("🔍 영단어 · 숙어 · 한영 정밀 검색")
    st.caption(
        "단어·숙어·활용형 또는 한국어 뜻을 입력하면 수능·모의고사 독해 기준으로 뜻·발음·품사·유의어·어원·예문을 정리해 드리며, "
        "한번 검색된 단어는 **공용 사전 DB에 영구 캐싱**되어 이후 재조회 시 AI 토큰이 전혀 들지 않습니다."
    )

    pending_q = str(st.session_state.pop("tab4_dict_pending_query", "") or "").strip()
    if pending_q:
        st.session_state["tab4_dict_search_input"] = pending_q

    with st.form("tab4_dict_search_form", border=True):
        sc1, sc2, sc3 = st.columns([2.8, 1.3, 0.9])
        with sc1:
            query_text = st.text_input(
                "검색어 (영단어 · 숙어 · 한국어 뜻)",
                placeholder="예: resilience, take into account, 모호한",
                key="tab4_dict_search_input",
            )
        with sc2:
            st.markdown("<div style='height: 1.55rem;'></div>", unsafe_allow_html=True)
            force_refresh = st.checkbox(
                "캐시 무시 · AI 재조회",
                value=False,
                key="tab4_dict_force_refresh",
            )
        with sc3:
            st.markdown("<div style='height: 1.45rem;'></div>", unsafe_allow_html=True)
            search_submitted = st.form_submit_button("🔍 검색", type="primary", width="stretch")

    trigger_q = pending_q or (query_text.strip() if search_submitted else "")
    if trigger_q:
        with st.spinner(f"📖 '{trigger_q}' 영단어 사전 조회 중..."):
            entry, origin, err = _execute_dict_lookup(
                ctx,
                trigger_q,
                force_refresh=(force_refresh and not pending_q),
            )
        if err:
            st.error(err)
        elif entry is not None:
            st.session_state["tab4_dict_current_entry"] = entry.model_dump()
            st.session_state["tab4_dict_current_meta"] = {
                "origin": origin,
                "raw_query": trigger_q,
            }

    cur_raw = st.session_state.get("tab4_dict_current_entry")
    cur_meta = st.session_state.get("tab4_dict_current_meta") or {}
    if isinstance(cur_raw, dict):
        try:
            cur_entry = EnglishDictEntry.model_validate(cur_raw)
            _render_dict_entry_card(
                ctx,
                cur_entry,
                origin_tag=str(cur_meta.get("origin") or "cache"),
                raw_query=str(cur_meta.get("raw_query") or cur_entry.headword),
            )
        except Exception:
            pass

    cached_total = ctx.storage.count_cached_english_dict_entries()
    with st.expander(f"📚 공용 사전 DB 빠른 찾아보기 (누적 {cached_total}개 · 클릭 시 토큰 0)", expanded=False):
        if cached_total == 0:
            st.info("아직 공용 사전 DB에 캐싱된 단어가 없습니다. 위 검색창에서 첫 단어를 검색해 보세요!")
        else:
            cache_filter = st.text_input(
                "캐시 내 단어 · 뜻 필터",
                placeholder="예: resi 또는 회복",
                key="tab4_dict_cache_filter",
            )
            cached_list = ctx.storage.list_cached_english_dict_entries(search=cache_filter, limit=40)
            if not cached_list:
                st.info("조건에 맞는 캐시 단어가 없습니다.")
            else:
                cols_per_row = 4
                for i in range(0, len(cached_list), cols_per_row):
                    row_items = cached_list[i : i + cols_per_row]
                    row_cols = st.columns(cols_per_row)
                    for ci, cent in enumerate(row_items):
                        with row_cols[ci]:
                            short_mean = (cent.summary_ko or "").strip()[:14]
                            lbl = f"{cent.headword} ({short_mean})" if short_mean else cent.headword
                            if st.button(lbl, key=f"tab4_cache_btn_{i}_{ci}", width="stretch"):
                                st.session_state["tab4_dict_pending_query"] = cent.headword
                                st.rerun()


def _collect_exam_vocab_rows(records: list[QuestionRecord]) -> list[dict[str, str]]:
    """저장된 영어 기출 문항의 key_vocabulary를 중복 제거하여 어휘 리스트로 반환한다."""
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for rec in records:
        if getattr(rec, "subject_category", "") != "영어":
            continue
        kv_list = list(getattr(rec, "key_vocabulary", []) or [])
        for raw_kv in kv_list:
            word, meaning = _parse_key_vocab_entry(raw_kv)
            norm = normalize_english_dict_key(word)
            if not norm or not re.search(r"[a-z]", norm):
                continue
            if norm in seen:
                continue
            seen.add(norm)
            rows.append(
                {
                    "headword": word,
                    "normalized_key": norm,
                    "meaning_ko": meaning,
                    "exam_name": rec.exam_name,
                    "q_num": str(rec.q_num),
                }
            )
    return rows


def _render_exam_vocab_subtab(ctx: AppContext) -> None:
    st.subheader("📝 기출 핵심 어휘 모아보기")
    st.caption(
        "**[탭 1]**에서 분석·저장한 영어 모의고사 지문의 핵심 어휘(`key_vocabulary`)를 모아보고, "
        "토큰 소모 없이 내 단어장에 바로 담거나 사전으로 상세 조회할 수 있습니다."
    )

    records = ctx.storage.load_db()
    eng_exams = sorted({r.exam_name for r in records if getattr(r, "subject_category", "") == "영어"})
    if not eng_exams:
        st.info("아직 DB에 저장된 영어 모의고사가 없습니다. **[탭 1]**에서 영어 시험지를 업로드해 주세요.")
        return

    fc1, fc2 = st.columns([1.8, 1.4])
    with fc1:
        sel_exams = st.multiselect(
            "시험 회차 필터 (비워두면 전체 회차)",
            options=eng_exams,
            default=[],
            key="tab4_exam_vocab_exams",
        )
    with fc2:
        kw_filter = st.text_input(
            "어휘 · 뜻 검색",
            placeholder="단어 또는 한국어 뜻",
            key="tab4_exam_vocab_kw",
        ).strip().lower()

    target_recs = [
        r for r in records
        if getattr(r, "subject_category", "") == "영어" and (not sel_exams or r.exam_name in sel_exams)
    ]
    vocab_rows = _collect_exam_vocab_rows(target_recs)
    if kw_filter:
        vocab_rows = [
            vr for vr in vocab_rows
            if kw_filter in vr["headword"].lower() or kw_filter in vr["meaning_ko"].lower()
        ]

    username = ctx.current_user.username if ctx.current_user else ""
    existing_keys = {v.normalized_key for v in ctx.storage.load_user_vocab(username)} if username else set()

    ac1, ac2 = st.columns([2.2, 1.3])
    with ac1:
        st.caption(f"조회된 기출 핵심 어휘: 총 **{len(vocab_rows)}개** (상위 80개 표시)")
    with ac2:
        if st.button(
            f"⭐ 필터된 어휘 전체 내 단어장에 담기 ({len(vocab_rows)}개 · 토큰 0)",
            disabled=(not vocab_rows or not username),
            type="primary",
            key="tab4_exam_vocab_batch_add",
            width="stretch",
        ):
            batch_items = [
                UserVocabItem(
                    headword=vr["headword"],
                    normalized_key=vr["normalized_key"],
                    summary_ko=vr["meaning_ko"] or "(뜻 확인 필요)",
                    source_exam=f"{vr['exam_name']} {vr['q_num']}번",
                )
                for vr in vocab_rows
            ]
            added_cnt, skip_cnt = ctx.storage.save_user_vocab_batch(username, batch_items)
            st.success(f"✅ {added_cnt}개 신규 저장 완료 (이미 저장된 {skip_cnt}개 제외)")
            st.rerun()

    if not vocab_rows:
        st.info("표시할 기출 어휘가 없습니다.")
        return

    for idx, vr in enumerate(vocab_rows[:80]):
        is_saved = vr["normalized_key"] in existing_keys
        rc1, rc2, rc3, rc4 = st.columns([1.6, 2.3, 0.75, 0.85])
        with rc1:
            st.markdown(f"**{vr['headword']}**")
        with rc2:
            st.markdown(f"{vr['meaning_ko'] or '-'} <span style='color:#64748B; font-size:0.8rem;'>({ _html.escape(vr['exam_name']) } {vr['q_num']}번)</span>", unsafe_allow_html=True)
        with rc3:
            if st.button("🔍 사전", key=f"tab4_ev_lookup_{idx}_{vr['normalized_key']}", width="stretch"):
                st.session_state["tab4_dict_pending_query"] = vr["headword"]
                st.toast(f"'{vr['headword']}' 검색이 준비되었습니다. [🔍 단어·숙어 검색] 탭을 확인해 주세요.")
        with rc4:
            if is_saved:
                st.caption("✅ 저장됨")
            else:
                if st.button("➕ 담기", key=f"tab4_ev_add_{idx}_{vr['normalized_key']}", width="stretch"):
                    ctx.storage.save_user_vocab_item(
                        username,
                        UserVocabItem(
                            headword=vr["headword"],
                            normalized_key=vr["normalized_key"],
                            summary_ko=vr["meaning_ko"] or "(뜻 확인 필요)",
                            source_exam=f"{vr['exam_name']} {vr['q_num']}번",
                        ),
                    )
                    st.toast(f"'{vr['headword']}' 저장 완료")
                    st.rerun()


def _render_user_wordbook_subtab(ctx: AppContext) -> None:
    st.subheader("⭐ 나만의 영단어장 & 암기장·테스트지 PDF")
    username = ctx.current_user.username if ctx.current_user else ""
    if not username:
        st.info("로그인 후 나만의 단어장을 이용할 수 있습니다.")
        return

    vocab_list = ctx.storage.load_user_vocab(username)
    total_cnt = len(vocab_list)
    mastered_cnt = sum(1 for v in vocab_list if v.mastered)
    unmastered_cnt = total_cnt - mastered_cnt
    wrong_cnt = sum(1 for v in vocab_list if v.wrong_count > 0)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("전체 저장 단어", f"{total_cnt}개")
    mc2.metric("미암기 단어", f"{unmastered_cnt}개")
    mc3.metric("암기 완료", f"{mastered_cnt}개")
    mc4.metric("오답 누적 단어", f"{wrong_cnt}개")

    with st.expander("➕ 단어 직접 추가 · 한꺼번에 붙여넣기 (토큰 0)", expanded=(total_cnt == 0)):
        add_c1, add_c2 = st.columns(2)
        with add_c1:
            with st.form("tab4_manual_single_add", clear_on_submit=True):
                st.markdown("**단어 1건 직접 추가**")
                m_word = st.text_input("영단어 · 숙어", placeholder="예: ubiquitous")
                m_mean = st.text_input("한국어 뜻", placeholder="예: 어디에나 있는, 아주 흔한")
                m_memo = st.text_input("메모 (선택)", placeholder="예: 6월 모평 31번 빈칸")
                if st.form_submit_button("💾 단어장에 추가", type="primary", width="stretch"):
                    nk = normalize_english_dict_key(m_word)
                    if not nk or not m_mean.strip():
                        st.error("영단어와 뜻을 모두 입력해 주세요.")
                    else:
                        ctx.storage.save_user_vocab_item(
                            username,
                            UserVocabItem(
                                headword=m_word.strip(),
                                normalized_key=nk,
                                summary_ko=m_mean.strip(),
                                memo=m_memo.strip(),
                                source_exam="직접 등록",
                            ),
                        )
                        st.success(f"✅ '{m_word.strip()}' 추가 완료!")
                        st.rerun()
        with add_c2:
            with st.form("tab4_manual_bulk_add", clear_on_submit=True):
                st.markdown("**여러 줄 한꺼번에 등록 (`단어 : 뜻` 형식)**")
                bulk_txt = st.text_area(
                    "목록 입력",
                    height=145,
                    placeholder="resilience : 회복탄력성\npragmatic : 실용적인\ntake into account : 고려하다",
                    label_visibility="collapsed",
                )
                if st.form_submit_button("📥 여러 단어 일괄 저장", width="stretch"):
                    bulk_items: list[UserVocabItem] = []
                    for line in bulk_txt.splitlines():
                        w, m = _parse_key_vocab_entry(line)
                        nk = normalize_english_dict_key(w)
                        if nk and m:
                            bulk_items.append(
                                UserVocabItem(
                                    headword=w,
                                    normalized_key=nk,
                                    summary_ko=m,
                                    source_exam="직접 일괄 등록",
                                )
                            )
                    if not bulk_items:
                        st.error("`'단어 : 뜻'` 형식의 유효한 줄을 입력해 주세요.")
                    else:
                        added, skipped = ctx.storage.save_user_vocab_batch(username, bulk_items)
                        st.success(f"✅ {added}개 저장 완료 (중복 {skipped}개 제외)")
                        st.rerun()

    if not vocab_list:
        st.info("단어장이 비어 있습니다. **[🔍 단어·숙어 검색]**이나 **[📝 기출 어휘 모아보기]**에서 단어를 저장해 보세요!")
        return

    fc1, fc2, fc3 = st.columns([1.2, 1.5, 1.0])
    with fc1:
        status_filter = st.selectbox(
            "암기 상태 필터",
            options=["전체", "미암기만", "암기 완료만", "오답 누적만"],
            key="tab4_wb_status_filter",
        )
    with fc2:
        wb_search = st.text_input(
            "단어장 내 검색",
            placeholder="영단어 · 뜻 · 메모",
            key="tab4_wb_search",
        ).strip().lower()
    with fc3:
        st.markdown("<div style='height: 1.55rem;'></div>", unsafe_allow_html=True)
        blind_mode = st.toggle("🙈 뜻 가리기", value=False, key="tab4_wb_blind_mode")

    filtered: list[UserVocabItem] = []
    for it in vocab_list:
        if status_filter == "미암기만" and it.mastered:
            continue
        if status_filter == "암기 완료만" and not it.mastered:
            continue
        if status_filter == "오답 누적만" and it.wrong_count <= 0:
            continue
        if wb_search:
            hay = f"{it.headword} {it.summary_ko} {it.memo} {it.source_exam}".lower()
            if wb_search not in hay:
                continue
        filtered.append(it)

    st.caption(f"표시 중: **{len(filtered)} / {total_cnt}개** (최대 100개 표시)")

    for idx, it in enumerate(filtered[:100]):
        with st.container(border=True):
            c_chk, c_word, c_mean, c_act = st.columns([0.55, 1.5, 2.35, 1.1])
            with c_chk:
                new_m = st.checkbox(
                    "암기",
                    value=bool(it.mastered),
                    key=f"tab4_wb_chk_{it.normalized_key}_{idx}",
                    label_visibility="collapsed",
                    help="체크하면 암기 완료로 표시됩니다.",
                )
                if new_m != bool(it.mastered):
                    ctx.storage.update_user_vocab_item(username, it.normalized_key, mastered=new_m)
                    st.rerun()
            with c_word:
                ipa = f" `{it.phonetic}`" if it.phonetic else ""
                wrong_badge = f" · 🔁 오답 {it.wrong_count}회" if it.wrong_count > 0 else ""
                st.markdown(f"**{it.headword}**{ipa}")
                if it.pos_summary or it.source_exam or wrong_badge:
                    st.caption(f"{it.pos_summary or '어휘'} · {it.source_exam or '사전'}{wrong_badge}")
            with c_mean:
                if blind_mode:
                    with st.expander("👀 뜻 · 예문 확인하기", expanded=False):
                        st.write(f"**{it.summary_ko}**")
                        if it.example_en:
                            st.caption(f"{it.example_en} ({it.example_ko})")
                else:
                    st.write(f"**{it.summary_ko}**")
                    if it.memo:
                        st.caption(f"📝 메모: {it.memo}")
            with c_act:
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("🔍", key=f"tab4_wb_go_{it.normalized_key}_{idx}", help="사전에서 상세 보기"):
                        st.session_state["tab4_dict_pending_query"] = it.headword
                        st.toast(f"'{it.headword}' 검색이 준비되었습니다. [🔍 단어·숙어 검색] 탭을 확인하세요.")
                with b2:
                    if st.button("🗑️", key=f"tab4_wb_del_{it.normalized_key}_{idx}", help="단어장에서 삭제"):
                        ctx.storage.delete_user_vocab_item(username, it.normalized_key)
                        st.rerun()

    # 단어장 / 테스트지 PDF 출력 & 일괄 정리
    st.markdown("---")
    with st.container(border=True):
        hdr_c1, hdr_c2 = st.columns([2.8, 1.3])
        with hdr_c1:
            st.markdown("#### 📄 내 단어장 · 암기 테스트지 A4 PDF 출력")
        with hdr_c2:
            if mastered_cnt > 0 and st.button(
                f"🧹 암기 완료 단어 비우기 ({mastered_cnt}개)",
                key="tab4_wb_clear_mastered_btn",
                width="stretch",
            ):
                removed = ctx.storage.clear_user_vocab(username, only_mastered=True)
                st.success(f"🧹 암기 완료 단어 {removed}개를 정리했습니다.")
                st.rerun()
        p1, p2, p3 = st.columns([1.2, 1.4, 1.3])
        with p1:
            pdf_scope = st.selectbox(
                "출력 대상",
                options=["현재 필터 단어", "미암기 단어만", "전체 단어"],
                key="tab4_pdf_scope",
            )
        with p2:
            pdf_mode_label = st.selectbox(
                "PDF 양식",
                options=[
                    "📖 암기용 단어장 (2단 정렬)",
                    "✏️ 뜻 쓰기 테스트지 + 정답지",
                    "🔤 영단어 스펠링 테스트지 + 정답지",
                ],
                key="tab4_pdf_mode",
            )
        with p3:
            pdf_title = st.text_input(
                "PDF 상단 제목",
                value="나만의 수능 영단어장",
                key="tab4_pdf_title",
            )

        o1, o2, _ = st.columns([1.0, 1.2, 1.4])
        with o1:
            pdf_shuffle = st.checkbox("단어 순서 랜덤 섞기", value=True, key="tab4_pdf_shuffle")
        with o2:
            first_letter_hint = st.checkbox("스펠링 첫 글자 힌트 (r____)", value=True, key="tab4_pdf_hint")

        if pdf_scope == "현재 필터 단어":
            target_pdf_items = list(filtered)
        elif pdf_scope == "미암기 단어만":
            target_pdf_items = [v for v in vocab_list if not v.mastered]
        else:
            target_pdf_items = list(vocab_list)

        mode_code = (
            "study"
            if pdf_mode_label.startswith("📖")
            else ("test_meaning" if pdf_mode_label.startswith("✏️") else "test_word")
        )

        if st.button(
            f"📥 PDF 생성 ({len(target_pdf_items)}단어)",
            disabled=(len(target_pdf_items) == 0),
            type="primary",
            key="tab4_gen_vocab_pdf_btn",
        ):
            ordered = list(target_pdf_items)
            if pdf_shuffle:
                random.shuffle(ordered)
            pdf_bytes = generate_vocab_wordbook_pdf(
                ordered,
                title=pdf_title.strip() or "나만의 수능 영단어장",
                mode=mode_code,
                include_first_letter_hint=first_letter_hint,
            )
            st.session_state["tab4_vocab_pdf_bytes"] = pdf_bytes
            st.session_state["tab4_vocab_pdf_name"] = f"{sanitize_id_part(pdf_title)}.pdf"
            st.success(f"✅ {len(ordered)}개 단어 PDF 생성 완료!")

        dl_pdf = st.session_state.get("tab4_vocab_pdf_bytes")
        if isinstance(dl_pdf, (bytes, bytearray)) and len(dl_pdf) > 0:
            st.download_button(
                label=f"⬇️ 영단어 PDF 다운로드 ({len(dl_pdf) // 1024} KB)",
                data=bytes(dl_pdf),
                file_name=str(st.session_state.get("tab4_vocab_pdf_name") or "vocab_wordbook.pdf"),
                mime="application/pdf",
                type="primary",
                width="stretch",
                key="tab4_dl_vocab_pdf_btn",
            )


def _load_quiz_pool(ctx: AppContext, source_choice: str, username: str) -> list[UserVocabItem]:
    if source_choice == "내 단어장 (미암기 우선)":
        all_v = ctx.storage.load_user_vocab(username) if username else []
        unmastered = [v for v in all_v if not v.mastered]
        return unmastered if unmastered else all_v
    if source_choice == "내 단어장 (전체)":
        return ctx.storage.load_user_vocab(username) if username else []
    if source_choice == "기출 핵심 어휘 DB":
        rows = _collect_exam_vocab_rows(ctx.storage.load_db())
        return [
            UserVocabItem(
                headword=r["headword"],
                normalized_key=r["normalized_key"],
                summary_ko=r["meaning_ko"],
                source_exam=f"{r['exam_name']} {r['q_num']}번",
            )
            for r in rows
            if r["meaning_ko"]
        ]
    cached_ents = ctx.storage.list_cached_english_dict_entries(limit=150)
    return [_entry_to_user_vocab_item(ent, source_exam="공용 사전 DB") for ent in cached_ents if ent.summary_ko]


def _rerun_quiz_fragment() -> None:
    """실제 브라우저에서는 프래그먼트 부분 리런을, 전체 리런 컨텍스트에서는 일반 리런을 수행한다."""
    try:
        st.rerun(scope="fragment")
    except Exception:
        st.rerun()


@st.fragment
def _render_vocab_quiz_fragment(ctx: AppContext) -> None:
    st.subheader("🎯 영단어 플래시카드 & 4지선다 암기 퀴즈")
    username = ctx.current_user.username if ctx.current_user else ""

    qc1, qc2, qc3 = st.columns([1.4, 1.2, 1.2])
    with qc1:
        quiz_source = st.selectbox(
            "출제 어휘 출처",
            options=[
                "내 단어장 (미암기 우선)",
                "내 단어장 (전체)",
                "기출 핵심 어휘 DB",
                "공용 사전 캐시 DB",
            ],
            key="tab4_quiz_source",
        )
    with qc2:
        quiz_mode = st.radio(
            "학습 모드",
            options=["🃏 플래시카드", "✏️ 4지선다 테스트"],
            horizontal=True,
            key="tab4_quiz_mode",
        )
    with qc3:
        quiz_dir = st.radio(
            "출제 방향",
            options=["영어 → 뜻", "뜻 → 영어"],
            horizontal=True,
            key="tab4_quiz_dir",
        )

    pool = _load_quiz_pool(ctx, quiz_source, username)
    if not pool:
        st.info("선택한 출처에 학습 가능한 단어가 없습니다. 먼저 단어를 검색·저장하거나 다른 출처를 선택해 주세요.")
        return

    user_saved_keys = {v.normalized_key for v in ctx.storage.load_user_vocab(username)} if username else set()

    if quiz_mode.startswith("🃏"):
        fc_state_sig = f"{quiz_source}||{len(pool)}"
        if st.session_state.get("tab4_fc_sig") != fc_state_sig:
            st.session_state["tab4_fc_sig"] = fc_state_sig
            st.session_state["tab4_fc_order"] = [it.normalized_key for it in pool]
            random.shuffle(st.session_state["tab4_fc_order"])
            st.session_state["tab4_fc_idx"] = 0
            st.session_state["tab4_fc_revealed"] = False

        pool_map = {it.normalized_key: it for it in pool}
        order_keys = [k for k in st.session_state.get("tab4_fc_order", []) if k in pool_map]
        if not order_keys:
            order_keys = list(pool_map.keys())
            st.session_state["tab4_fc_order"] = order_keys

        fc_idx = int(st.session_state.get("tab4_fc_idx", 0)) % len(order_keys)
        cur_item = pool_map[order_keys[fc_idx]]
        revealed = bool(st.session_state.get("tab4_fc_revealed", False))

        front_txt = cur_item.headword if quiz_dir == "영어 → 뜻" else cur_item.summary_ko
        back_txt = cur_item.summary_ko if quiz_dir == "영어 → 뜻" else cur_item.headword

        with st.container(border=True):
            st.caption(f"카드 **{fc_idx + 1} / {len(order_keys)}** · 출처: {cur_item.source_exam or quiz_source}")
            st.markdown(
                f"<div style='text-align:center; padding:1.4rem 1rem; background:#F8FAFC; border:1px solid #CBD5E1; "
                f"border-radius:12px; margin:0.4rem 0 0.8rem 0;'>"
                f"<div style='font-size:1.7rem; font-weight:800; color:#0F172A;'>{_html.escape(front_txt)}</div>"
                f"<div style='font-size:0.9rem; color:#64748B; margin-top:4px;'>{_html.escape(cur_item.phonetic if quiz_dir == '영어 → 뜻' else cur_item.pos_summary)}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if quiz_dir == "영어 → 뜻" or revealed:
                _render_tts_buttons(cur_item.headword)

            if revealed:
                ex_line = f"\n\n> *{cur_item.example_en}*\n> {cur_item.example_ko}" if cur_item.example_en else ""
                st.success(f"**정답:** {back_txt}{ex_line}")

            bc1, bc2, bc3, bc4 = st.columns(4)
            with bc1:
                if st.button("👀 뒤집기 / 정답 확인", key="tab4_fc_flip", width="stretch"):
                    st.session_state["tab4_fc_revealed"] = not revealed
                    _rerun_quiz_fragment()
            with bc2:
                if st.button("✅ 외웠어요", type="primary", key="tab4_fc_ok", width="stretch"):
                    if username and cur_item.normalized_key in user_saved_keys:
                        ctx.storage.update_user_vocab_item(username, cur_item.normalized_key, mastered=True)
                    st.session_state["tab4_fc_idx"] = (fc_idx + 1) % len(order_keys)
                    st.session_state["tab4_fc_revealed"] = False
                    _rerun_quiz_fragment()
            with bc3:
                if st.button("🔁 다시 볼래요", key="tab4_fc_retry", width="stretch"):
                    if username and cur_item.normalized_key in user_saved_keys:
                        ctx.storage.update_user_vocab_item(
                            username,
                            cur_item.normalized_key,
                            mastered=False,
                            increment_wrong=True,
                        )
                    st.session_state["tab4_fc_idx"] = (fc_idx + 1) % len(order_keys)
                    st.session_state["tab4_fc_revealed"] = False
                    _rerun_quiz_fragment()
            with bc4:
                if st.button("🔀 순서 새로 섞기", key="tab4_fc_shuffle", width="stretch"):
                    random.shuffle(st.session_state["tab4_fc_order"])
                    st.session_state["tab4_fc_idx"] = 0
                    st.session_state["tab4_fc_revealed"] = False
                    _rerun_quiz_fragment()
        return

    # 4지선다 객관식 테스트 모드
    max_q_cnt = min(20, len(pool))
    tc1, tc2 = st.columns([1.4, 1.0])
    with tc1:
        n_quiz = int(
            st.number_input(
                "문항 수",
                min_value=1,
                max_value=max(1, max_q_cnt),
                value=min(10, max_q_cnt),
                step=1,
                key="tab4_mc_count",
            )
        )
    with tc2:
        st.markdown("<div style='height: 1.55rem;'></div>", unsafe_allow_html=True)
        start_quiz_btn = st.button("🎲 새 4지선다 시험지 생성", type="primary", key="tab4_mc_gen_btn", width="stretch")

    if (
        start_quiz_btn
        or "tab4_mc_questions" not in st.session_state
        or st.session_state.get("tab4_mc_dir") != quiz_dir
    ):
        picked = random.sample(pool, k=min(n_quiz, len(pool)))
        # 오답 선지 풀 (pool + 캐시 + 기본 수능 어휘 뱅크)
        distractor_pool: list[tuple[str, str]] = [(p.headword, p.summary_ko) for p in pool if p.summary_ko]
        for fb_w, fb_m in _FALLBACK_DISTRACTOR_BANK:
            distractor_pool.append((fb_w, fb_m))

        questions: list[dict[str, Any]] = []
        for idx, target in enumerate(picked):
            prompt_str = target.headword if quiz_dir == "영어 → 뜻" else target.summary_ko
            ans_str = target.summary_ko if quiz_dir == "영어 → 뜻" else target.headword
            distractors: list[str] = []
            shuffled_cand = list(distractor_pool)
            random.shuffle(shuffled_cand)
            for cw, cm in shuffled_cand:
                cand_val = cm if quiz_dir == "영어 → 뜻" else cw
                if cand_val and cand_val != ans_str and cand_val not in distractors:
                    distractors.append(cand_val)
                if len(distractors) >= 3:
                    break
            options = [ans_str, *distractors[:3]]
            random.shuffle(options)
            questions.append(
                {
                    "idx": idx + 1,
                    "item": target.model_dump(),
                    "prompt": prompt_str,
                    "answer": ans_str,
                    "options": options,
                }
            )
        st.session_state["tab4_mc_gen_id"] = int(st.session_state.get("tab4_mc_gen_id", 0)) + 1
        st.session_state["tab4_mc_dir"] = quiz_dir
        st.session_state["tab4_mc_questions"] = questions
        st.session_state["tab4_mc_graded"] = False
        st.session_state["tab4_mc_picks"] = {}
        if start_quiz_btn:
            _rerun_quiz_fragment()

    questions = st.session_state.get("tab4_mc_questions") or []
    if not questions:
        return

    gen_id = int(st.session_state.get("tab4_mc_gen_id", 1))
    graded = bool(st.session_state.get("tab4_mc_graded", False))
    user_picks: dict[int, str] = st.session_state.get("tab4_mc_picks") or {}

    if graded:
        correct_cnt = sum(1 for q in questions if user_picks.get(q["idx"]) == q["answer"])
        pct = round((correct_cnt / max(1, len(questions))) * 100, 1)
        st.markdown(
            f"<div class='score-report-card'>"
            f"<div class='score-report-top'><div class='score-big'>🏆 영단어 퀴즈 결과: {correct_cnt} / {len(questions)} ({pct}점)</div></div>"
            f"<div class='score-bar-track'><div class='score-bar-fill' style='width:{pct}%;'></div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    for q in questions:
        qi = q["idx"]
        with st.container(border=True):
            st.markdown(f"**Q{qi}. `{q['prompt']}`** 의 알맞은 {'뜻' if quiz_dir == '영어 → 뜻' else '영단어'}은?")
            choice = st.radio(
                f"Q{qi} 선택",
                options=q["options"],
                index=None if qi not in user_picks else (q["options"].index(user_picks[qi]) if user_picks[qi] in q["options"] else None),
                key=f"tab4_mc_radio_{gen_id}_{qi}",
                label_visibility="collapsed",
            )
            if choice is not None:
                user_picks[qi] = choice
                st.session_state["tab4_mc_picks"] = user_picks
            if graded:
                if user_picks.get(qi) == q["answer"]:
                    st.success(f"⭕ 정답! ({q['answer']})")
                else:
                    st.error(f"❌ 오답 · 정답: **{q['answer']}**")

    g1, g2 = st.columns(2)
    with g1:
        if st.button("📝 채점하기", type="primary", key="tab4_mc_grade_btn", width="stretch"):
            st.session_state["tab4_mc_graded"] = True
            if username:
                for q in questions:
                    if user_picks.get(q["idx"]) != q["answer"]:
                        nk = str(q["item"].get("normalized_key") or "")
                        if nk in user_saved_keys:
                            ctx.storage.update_user_vocab_item(username, nk, mastered=False, increment_wrong=True)
            _rerun_quiz_fragment()
    with g2:
        if graded and username:
            wrong_items = [
                UserVocabItem.model_validate(q["item"])
                for q in questions
                if user_picks.get(q["idx"]) != q["answer"]
            ]
            if wrong_items and st.button(
                f"⭐ 틀린 단어 {len(wrong_items)}개 내 단어장에 추가",
                key="tab4_mc_save_wrong_btn",
                width="stretch",
            ):
                added, skipped = ctx.storage.save_user_vocab_batch(username, wrong_items)
                st.success(f"✅ 오답 단어 {added}개 저장 완료 (기존 {skipped}개)")


def render_tab4(ctx: AppContext) -> None:
    """[탭 4] 올인원 수능 영단어 사전 · 기출 어휘 연동 · 개인 단어장 & PDF · 암기 퀴즈 UI."""
    sub_search, sub_exam_vocab, sub_wordbook, sub_quiz = st.tabs(
        [
            "🔍 단어 · 숙어 검색",
            "📝 기출 어휘 모아보기",
            "⭐ 나만의 단어장 & PDF",
            "🎯 영단어 암기 퀴즈",
        ]
    )

    with sub_search:
        _render_dict_search_subtab(ctx)

    with sub_exam_vocab:
        _render_exam_vocab_subtab(ctx)

    with sub_wordbook:
        _render_user_wordbook_subtab(ctx)

    with sub_quiz:
        _render_vocab_quiz_fragment(ctx)
