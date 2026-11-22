from __future__ import annotations

import html as _html
import json
import re
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from src.auth import resolve_active_gemini_api_key
from src.config import FLASH_LITE_MODEL
from src.models import (
    EnglishDictEntry,
    QuestionRecord,
    UserVocabItem,
    normalize_english_dict_key,
)
from src.ui.common import (
    create_lazy_expander,
    create_lazy_tabs,
    ensure_token_quota,
    is_container_open,
    sanitize_id_part,
    usage_scope,
)
from src.ui.context import AppContext
from src.ui_styles import render_chips

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

    records = ctx.storage.load_db_readonly()
    ctx_hits = find_word_in_exam_passages(records, q, raw_query=q, max_hits=2)
    ctx_sentences = [h["sentence"] for h in ctx_hits if h.get("sentence")]

    try:
        with usage_scope(ctx):
            from src.gemini.dictionary import lookup_english_word_with_gemini

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

    records = ctx.storage.load_db_readonly()
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
    cache_exp = create_lazy_expander(
        f"📚 공용 사전 DB 빠른 찾아보기 (누적 {cached_total}개 · 클릭 시 토큰 0)",
        expanded=False,
        key="tab4_dict_cache_exp",
    )
    with cache_exp:
        if not is_container_open(cache_exp):
            pass
        elif cached_total == 0:
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

    records = ctx.storage.load_db_readonly()
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

    ev_page_size = 20
    ev_total_pages = max(1, (len(vocab_rows) + ev_page_size - 1) // ev_page_size)
    ac1, ac_pg, ac2 = st.columns([1.8, 0.9, 1.4])
    with ac1:
        st.caption(f"조회된 기출 핵심 어휘: 총 **{len(vocab_rows)}개** ({ev_page_size}개씩 페이지 표시)")
    with ac_pg:
        ev_page = int(
            st.number_input(
                "페이지",
                min_value=1,
                max_value=ev_total_pages,
                value=1,
                step=1,
                key="tab4_ev_page",
            )
        )
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

    ev_start = (ev_page - 1) * ev_page_size
    ev_slice = vocab_rows[ev_start : ev_start + ev_page_size]
    for offset, vr in enumerate(ev_slice):
        idx = ev_start + offset
        is_saved = vr["normalized_key"] in existing_keys
        rc1, rc2, rc3, rc4 = st.columns([1.6, 2.3, 0.75, 0.85])
        with rc1:
            st.markdown(f"**{vr['headword']}**")
        with rc2:
            st.markdown(f"{vr['meaning_ko'] or '-'} <span style='color:#64748B; font-size:0.8rem;'>({ _html.escape(vr['exam_name']) } {vr['q_num']}번)</span>", unsafe_allow_html=True)
        with rc3:
            if st.button("🔍 사전", key=f"tab4_ev_lookup_{idx}_{vr['normalized_key']}", width="stretch"):
                st.session_state["tab4_dict_pending_query"] = vr["headword"]
                st.session_state["tab4_switch_to_search"] = True
                st.rerun()
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

def render_tab4(ctx: AppContext) -> None:
    """[탭 4] 올인원 수능 영단어 사전 · 기출 어휘 연동 · 개인 단어장 & PDF · 암기 퀴즈 UI."""
    if st.session_state.pop("tab4_switch_to_search", False):
        st.session_state["tab4_dict_subtabs"] = "🔍 단어 · 숙어 검색"

    sub_search, sub_exam_vocab, sub_wordbook, sub_quiz = create_lazy_tabs(
        [
            "🔍 단어 · 숙어 검색",
            "📝 기출 어휘 모아보기",
            "⭐ 나만의 단어장 & PDF",
            "🎯 영단어 암기 퀴즈",
        ],
        key="tab4_dict_subtabs",
    )

    with sub_search:
        if is_container_open(sub_search):
            _render_dict_search_subtab(ctx)

    with sub_exam_vocab:
        if is_container_open(sub_exam_vocab):
            _render_exam_vocab_subtab(ctx)

    with sub_wordbook:
        if is_container_open(sub_wordbook):
            from src.ui.tab4_vocab_quiz import _render_user_wordbook_subtab

            _render_user_wordbook_subtab(ctx)

    with sub_quiz:
        if is_container_open(sub_quiz):
            from src.ui.tab4_vocab_quiz import _render_vocab_quiz_fragment

            _render_vocab_quiz_fragment(ctx)
