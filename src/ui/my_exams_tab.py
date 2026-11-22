from __future__ import annotations

import streamlit as st

from src.ui.common import deserialize_matched_items, sanitize_id_part
from src.ui.context import AppContext


def _render_my_exams_tab(ctx: AppContext) -> None:
    st.subheader("📁 나의 시험지 보관함 (과목별 자동 저장)")
    st.caption(
        "맞춤형 시험지 PDF를 생성하거나 웹에서 직접 풀고 채점한 시험지는 계정별로 과목에 맞춰 자동 저장됩니다."
    )
    if ctx.current_user is None:
        st.info("로그인 후 이용할 수 있습니다.")
        return

    all_exams = [
        ex
        for ex in ctx.storage.list_user_exams(ctx.current_user.username)
        if ex.get("exam_kind") != "korean_analysis"
    ]
    if not all_exams:
        st.info("아직 저장된 시험지가 없습니다. **탭 2**에서 PDF를 만들거나 문제를 풀고 채점하면 이곳에 자동 저장됩니다.")
        return

    subj_counts: dict[str, int] = {}
    for ex in all_exams:
        s = str(ex.get("subject") or "기타")
        subj_counts[s] = subj_counts.get(s, 0) + 1

    subj_options = ["전체"] + sorted(subj_counts.keys())
    sel_subj = st.radio(
        "과목별 보기",
        options=subj_options,
        format_func=lambda s: f"{s} ({len(all_exams)}건)" if s == "전체" else f"{s} ({subj_counts.get(s, 0)}건)",
        horizontal=True,
        key="my_exams_subject_filter",
    )

    shown_exams = [e for e in all_exams if sel_subj == "전체" or str(e.get("subject") or "기타") == sel_subj]
    if not shown_exams:
        st.info(f"'{sel_subj}' 과목에 저장된 시험지가 없습니다.")
        return

    for ex in shown_exams:
        ex_id = str(ex.get("exam_id") or "")
        ex_title = str(ex.get("title") or "맞춤형 시험지")
        ex_subj = str(ex.get("subject") or "기타")
        ex_q_cnt = int(ex.get("question_count") or len(ex.get("items") or []))
        ex_updated = str(ex.get("updated_at") or ex.get("created_at") or "")
        last_score = ex.get("last_score")

        with st.container(border=True):
            top_c1, top_c2 = st.columns([4, 1.5])
            with top_c1:
                st.markdown(f"**[{ex_subj}] {ex_title}**")
                count_label = f"총 {ex_q_cnt}문항"
                meta_parts = [count_label, f"저장일시: {ex_updated}"]
                if isinstance(last_score, dict):
                    meta_parts.append(
                        f"최근 점수: {last_score.get('percent', 0)}점 ({last_score.get('correct', 0)}/{last_score.get('total', ex_q_cnt)})"
                    )
                st.caption(" · ".join(meta_parts))
            with top_c2:
                pdf_data = ctx.storage.get_user_exam_pdf_bytes(ctx.current_user.username, ex_id)
                if pdf_data:
                    st.download_button(
                        label=f"⬇️ PDF ({len(pdf_data) // 1024}KB)",
                        data=pdf_data,
                        file_name=f"{sanitize_id_part(ex_title)}.pdf",
                        mime="application/pdf",
                        key=f"my_exam_dl_{ex_id}",
                        width="stretch",
                    )
                else:
                    st.caption("📄 PDF 미생성 (풀이 전용)")

            b_c1, b_c2, b_c3 = st.columns([1.4, 1.2, 3.4])
            with b_c1:
                if st.button("✏️ 이 시험지 다시 풀기", key=f"my_exam_load_{ex_id}", width="stretch"):
                    restored = deserialize_matched_items(ex.get("items") or [], storage=ctx.storage)
                    if restored:
                        st.session_state["matched_items"] = restored
                        st.session_state["user_answers"] = {}
                        st.session_state["graded"] = False
                        st.session_state["ai_explanations"] = {}
                        st.session_state["ai_chats"] = {}
                        st.session_state["generated_exam_pdf"] = pdf_data
                        st.session_state["generated_exam_pdf_sig"] = ""
                        st.success("시험지를 불러왔습니다! 상단 **[탭 2]**로 이동해 바로 풀거나 PDF를 생성하세요.")
                    else:
                        st.warning("시험지 문항 데이터를 복원하지 못했습니다.")
            with b_c2:
                if st.button("🗑️ 삭제", key=f"my_exam_del_{ex_id}", width="stretch"):
                    ctx.storage.delete_user_exam(ctx.current_user.username, ex_id)
                    st.rerun()




render_my_exams_tab = _render_my_exams_tab
