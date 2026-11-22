from __future__ import annotations

import html as _html
import random
from typing import Any

import streamlit as st

from src.models import UserVocabItem, normalize_english_dict_key
from src.ui.common import sanitize_id_part
from src.ui.context import AppContext
from src.ui.tab4_dictionary import (
    _collect_exam_vocab_rows,
    _entry_to_user_vocab_item,
    _parse_key_vocab_entry,
)
from src.ui_styles import render_chips

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

    wb_page_size = 25
    wb_total_pages = max(1, (len(filtered) + wb_page_size - 1) // wb_page_size)
    wb_c1, wb_c2 = st.columns([2.6, 0.9])
    with wb_c1:
        st.caption(f"표시 중: **{len(filtered)} / {total_cnt}개** ({wb_page_size}개씩 페이지 표시)")
    with wb_c2:
        wb_page = int(
            st.number_input(
                "단어장 페이지",
                min_value=1,
                max_value=wb_total_pages,
                value=1,
                step=1,
                key="tab4_wb_page",
            )
        )

    wb_start = (wb_page - 1) * wb_page_size
    for offset, it in enumerate(filtered[wb_start : wb_start + wb_page_size]):
        idx = wb_start + offset
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
                        st.session_state["tab4_switch_to_search"] = True
                        st.rerun()
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
            from src.pdf.vocab_pdf import generate_vocab_wordbook_pdf

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
                from src.ui.tab4_dictionary import _render_tts_buttons

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

