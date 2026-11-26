from __future__ import annotations

import contextlib
import hashlib
import html as _html
import json
import re
import threading
from typing import TYPE_CHECKING, Any, Sequence

import streamlit as st

from src.auth import (
    CreditReservation,
    acquire_credit_reservation,
    check_credit_quota,
    check_token_quota,
    consume_premium_model_usage,
    format_credit_count,
    get_bonus_credits,
    get_monthly_credit_limit,
    get_remaining_monthly_base_credits,
    get_total_available_credits,
    get_user_by_username,
    is_premium_model,
    is_server_api_key_user,
    record_token_usage,
    resolve_model_for_user,
    settle_credit_reservation,
)
from src.gemini.usage import TokenUsageRecord, bind_usage_tracker, has_active_usage_tracker
from src.ui.context import AppContext

if TYPE_CHECKING:
    from src.models import QuestionMatchJudgment, QuestionRecord, RewrittenQuestion
    from src.storage import StorageManager

CIRCLED_MAP = {"1": "①", "2": "②", "3": "③", "4": "④", "5": "⑤"}
REVERSE_CIRCLED_MAP = {v: k for k, v in CIRCLED_MAP.items()}


def normalize_answer(ans: str) -> str:
    s = (ans or "").strip().replace(" ", "")
    if s in REVERSE_CIRCLED_MAP:
        return REVERSE_CIRCLED_MAP[s]
    if len(s) == 2 and s[0] in CIRCLED_MAP and s[1] == "번":
        return s[0]
    return s.lower()


def is_multiple_choice_question(q: QuestionRecord, rewritten: RewrittenQuestion | None) -> bool:
    if rewritten and len(rewritten.rewritten_choices) >= 2:
        return True
    return normalize_answer(q.answer) in {"1", "2", "3", "4", "5"}


def sanitize_id_part(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣_\-]", "_", text.strip())
    return re.sub(r"_+", "_", cleaned).strip("_") or "exam"


def _init_session_defaults() -> None:
    defaults: dict[str, Any] = {
        "parsed_crops": [],
        "parsed_exam_meta": {},
        "ebsi_problem_bytes": None,
        "ebsi_answer_bytes": None,
        "ebsi_problem_name": "",
        "ebsi_answer_name": "",
        "concept_whitelist": None,
        "concept_sliced_text": "",
        "concept_source_label": "",
        "concept_pdf_hash": "",
        "matched_items": [],
        "match_diagnostics": [],
        "generated_exam_pdf": None,
        "user_answers": {},
        "graded": False,
        "ai_explanations": {},
        "ai_chats": {},
        "logged_in_username": "",
        "pending_google_signup": None,
        "auth_flash_msg": None,
        "open_payment_modal_requested": False,
        "_upg_cta_seq": 0,
        "tab1_last_loaded_sig": "",
        "tab1_meta_status": "",
        "meta_year": "",
        "meta_month": "",
        "meta_grade": "",
        "meta_subject": "",
        "meta_title": "",
        "tab1_subject_cat": "과학",
        "stage2_exam_name": "",
        "stage2_saved_q_nums": [],
        "stage2_failed_q_nums": [],
        "stage2_failed_errors": [],
        "stage2_sync_err": "",
        "trigger_retry_failed": False,
        "db_search_submitted": False,
        "db_search_filters": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v




init_session_defaults = _init_session_defaults


@st.cache_resource(ttl=1800, show_spinner=False)
def _get_cached_storage(local_root: str) -> StorageManager:
    from src.storage import StorageManager

    return StorageManager(local_root=local_root)




get_cached_storage = _get_cached_storage


@contextlib.contextmanager
def _usage_scope(
    ctx: AppContext | None = None,
    *,
    storage_obj: StorageManager | None = None,
    username: str | None = None,
    used_server_key: bool | None = None,
):
    """현재 로그인 사용자의 실제 Gemini API 토큰·USD 비용을 스레드 안전하게 누적해 종료 시 1회 저장한다."""
    if has_active_usage_tracker():
        yield
        return

    st_mgr = storage_obj or (ctx.storage if ctx is not None else _get_cached_storage(local_root="./exam_db"))
    uname = (
        username if username is not None else st.session_state.get("logged_in_username", "")
    ) or ""
    if not st_mgr or not uname:
        yield
        return

    if used_server_key is None:
        used_server_key = True

    acc_lock = threading.Lock()
    acc = {
        "prompt": 0,
        "cached": 0,
        "output": 0,
        "total": 0,
        "calls": 0,
        "cost_usd": 0.0,
    }

    def _on_usage(rec: TokenUsageRecord) -> None:
        with acc_lock:
            acc["prompt"] += rec.prompt_tokens
            acc["cached"] += rec.cached_tokens
            acc["output"] += rec.output_tokens
            acc["total"] += rec.total_tokens
            acc["calls"] += 1
            acc["cost_usd"] += rec.cost_usd

    with bind_usage_tracker(_on_usage):
        try:
            yield
        finally:
            if acc["total"] > 0 or acc["calls"] > 0 or acc["cost_usd"] > 0.0:
                try:
                    record_token_usage(
                        storage=st_mgr,
                        username=uname,
                        prompt_tokens=acc["prompt"],
                        cached_tokens=acc["cached"],
                        output_tokens=acc["output"],
                        total_tokens=acc["total"],
                        cost_usd=acc["cost_usd"],
                        api_calls=acc["calls"],
                        used_server_key=bool(used_server_key),
                        sync_drive=True,
                    )
                except Exception:
                    pass


usage_scope = _usage_scope


def _ensure_token_quota(
    ctx: AppContext | None = None,
    *,
    storage_obj: StorageManager | None = None,
    username: str | None = None,
    show_error: bool = True,
) -> bool:
    """API 호출 직전 계정 유효성을 확인한다 (0C 무료 기능/호환 래퍼)."""
    st_mgr = storage_obj or (ctx.storage if ctx is not None else _get_cached_storage(local_root="./exam_db"))
    uname = (
        username if username is not None else st.session_state.get("logged_in_username", "")
    ) or ""
    if not st_mgr or not uname:
        return True
    allowed, _rem, _limit, msg = check_token_quota(st_mgr, uname)
    if not allowed:
        if show_error and msg:
            st.error(msg)
        return False
    return True


ensure_token_quota = _ensure_token_quota


def _render_upgrade_cta_button(prefix: str = "upg_cta") -> None:
    """크레딧 부족 경고 아래에서 결제 신청 모달을 바로 여는 CTA 버튼을 렌더링한다."""
    uname = str(st.session_state.get("logged_in_username") or "").strip()
    if not uname:
        return
    seq = int(st.session_state.get("_upg_cta_seq", 0) or 0) + 1
    st.session_state["_upg_cta_seq"] = seq
    if st.button(
        "💎 멤버십 업그레이드 · 입금 신청하기",
        key=f"{prefix}_{seq}",
        type="primary",
    ):
        st.session_state["open_payment_modal_requested"] = True
        st.rerun()


def ensure_credit_quota(
    ctx: AppContext | None = None,
    required_credits: int = 1,
    action_label: str = "",
    *,
    storage_obj: StorageManager | None = None,
    username: str | None = None,
    show_error: bool = True,
) -> bool:
    """유료 기능 실행 전 가용 크레딧 충족 여부를 사전 검사한다."""
    st_mgr = storage_obj or (ctx.storage if ctx is not None else _get_cached_storage(local_root="./exam_db"))
    uname = (
        username if username is not None else st.session_state.get("logged_in_username", "")
    ) or ""
    if not st_mgr or not uname:
        return True
    allowed, _avail, _limit, msg = check_credit_quota(
        st_mgr,
        uname,
        required_credits=required_credits,
        action_label=action_label,
    )
    if not allowed:
        if show_error and msg:
            st.error(msg)
            if ctx is None or not ctx.is_admin:
                _render_upgrade_cta_button("ensure_upg_cta")
        return False
    return True


def try_reserve_credits(
    ctx: AppContext,
    reserve_credits: int,
    reason: str = "",
    *,
    show_error: bool = True,
) -> tuple[bool, CreditReservation | None]:
    """동일 유저 동시 실행을 차단하고 필요 크레딧을 원자적으로 선예치한다."""
    if ctx.current_user is None:
        if show_error:
            st.error("로그인이 필요합니다.")
        return False, None
    ok, reservation, msg = acquire_credit_reservation(
        ctx.storage,
        ctx.current_user.username,
        reserve_credits=reserve_credits,
        reason=reason,
    )
    if not ok and show_error and msg:
        st.error(msg)
        if not ctx.is_admin and "크레딧이 부족합니다" in msg:
            _render_upgrade_cta_button("reserve_upg_cta")
    return ok, reservation


def settle_credits(
    ctx: AppContext,
    reservation: CreditReservation | None,
    actual_credits: int,
    detail_note: str = "",
) -> None:
    """선예치된 크레딧을 최종 성공 건수 기준으로 정산하고 미사용분을 자동 환불한다."""
    settle_credit_reservation(
        ctx.storage,
        reservation,
        actual_credits=actual_credits,
        detail_note=detail_note,
    )


def format_credit_balance_inline(
    storage: StorageManager,
    username: str,
    charged: int | None = None,
) -> str:
    """완료 안내 배너에 붙일 차감 크레딧 및 잔여 크레딧 문자열을 생성한다."""
    u = get_user_by_username(storage, username) if username else None
    if u is None:
        return ""
    charged_part = f"차감 {format_credit_count(charged)}" if charged is not None else ""
    if u.tier == "ADMIN":
        return f" | {charged_part} (ADMIN 무제한)" if charged_part else ""
    base_lim = get_monthly_credit_limit(u) or 0
    base_rem = get_remaining_monthly_base_credits(u) or 0
    bonus = get_bonus_credits(u)
    avail = get_total_available_credits(u) or 0
    bonus_part = f" + 보너스 {bonus:,}C" if bonus > 0 else ""
    bal_part = f"가용 잔액 {avail:,}C (월간 {base_rem:,}/{base_lim:,}C{bonus_part})"
    if charged_part:
        return f" | {charged_part} · {bal_part}"
    return f" | {bal_part}"


@st.cache_data(ttl=300, show_spinner=False)
def _build_cached_exam_exports(
    _storage: StorageManager,
    exam_name: str,
    subj_cat: str,
    records_sig: str,
    _exam_recs: list[QuestionRecord],
) -> tuple[bytes, bytes, bytes, bytes]:
    """회차별 HTML/MD/JSONL/JSON 내보내기 바이트를 캐싱한다. 로컬 정리 파일의 mtime이 db.json 이상이면 즉시 재사용한다."""
    try:
        exam_dir = _storage._local_exam_dir(subj_cat, exam_name)
        html_p = exam_dir / "기출정리.html"
        md_p = exam_dir / "기출정리.md"
        jsonl_p = exam_dir / "ai_index.jsonl"
        json_p = exam_dir / "문제목록.json"
        if html_p.exists() and md_p.exists() and jsonl_p.exists() and json_p.exists():
            db_mtime = _storage.db_path.stat().st_mtime_ns if _storage.db_path.exists() else 0
            min_export_mtime = min(
                html_p.stat().st_mtime_ns,
                md_p.stat().st_mtime_ns,
                jsonl_p.stat().st_mtime_ns,
                json_p.stat().st_mtime_ns,
            )
            if min_export_mtime >= db_mtime:
                raw_json_bytes = json_p.read_bytes()
                parsed_list = json.loads(raw_json_bytes.decode("utf-8"))
                if isinstance(parsed_list, list) and len(parsed_list) == len(_exam_recs):
                    return (
                        html_p.read_bytes(),
                        md_p.read_bytes(),
                        jsonl_p.read_bytes(),
                        raw_json_bytes,
                    )
    except Exception:
        pass

    from src.exporters import build_exam_ai_jsonl, build_exam_html, build_exam_markdown

    html_str = build_exam_html(
        exam_name,
        subj_cat,
        _exam_recs,
        crop_loader=lambda fn, s=subj_cat, e=exam_name: _storage.get_crop_bytes(
            fn, subject_category=s, exam_name=e
        ),
    )
    md_str = build_exam_markdown(exam_name, subj_cat, _exam_recs)
    ai_jsonl_str = build_exam_ai_jsonl(_exam_recs)
    json_str = json.dumps(
        [r.model_dump() for r in _exam_recs],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        html_str.encode("utf-8"),
        md_str.encode("utf-8"),
        ai_jsonl_str.encode("utf-8"),
        json_str.encode("utf-8"),
    )




build_cached_exam_exports = _build_cached_exam_exports


def _render_english_rewritten_card_html(rw: RewrittenQuestion) -> str:
    """영어 수능형 변형 문항의 밑줄(<u>)·빈칸·[주어진 문장]·(A)-(B)-(C) 순서·[요약문] 박스 HTML을 조립한다."""
    from src.text_formatter import render_english_inline_html, split_english_stem_blocks

    blocks = split_english_stem_blocks(rw.rewritten_stem)
    body_parts: list[str] = []
    for kind, content in blocks:
        if kind == "instruction":
            body_parts.append(f"<div class='eng-instruction'>{render_english_inline_html(str(content))}</div>")
        elif kind == "given_box":
            g_title = str(content.get("title") or "주어진 글") if isinstance(content, dict) else "주어진 글"
            g_body = str(content.get("body") or "") if isinstance(content, dict) else str(content)
            body_parts.append(
                f"<div class='eng-given-box'><div class='eng-given-title'>[{_html.escape(g_title)}]</div>"
                f"<div>{render_english_inline_html(g_body)}</div></div>"
            )
        elif kind == "order_chunks":
            order_rows = "".join(
                f"<div class='eng-order-item'><span class='eng-order-mk'>{_html.escape(mk)}</span>"
                f"<span>{render_english_inline_html(txt)}</span></div>"
                for mk, txt in content
            )
            body_parts.append(f"<div class='eng-order-wrap'>{order_rows}</div>")
        elif kind == "summary_box":
            body_parts.append(
                f"<div class='eng-summary-arrow'>↓</div>"
                f"<div class='eng-summary-box'>{render_english_inline_html(str(content))}</div>"
            )
        elif kind == "footnote":
            body_parts.append(f"<div class='eng-footnote'>{render_english_inline_html(str(content))}</div>")
        else:
            body_parts.append(f"<div class='eng-passage'>{render_english_inline_html(str(content))}</div>")

    choices_html = ""
    if rw.rewritten_choices:
        choice_divs = []
        for ci, c_text in enumerate(rw.rewritten_choices):
            sym = CIRCLED_MAP.get(str(ci + 1), f"{ci + 1})")
            cleaned = (c_text or "").strip()
            if cleaned == sym or not cleaned:
                choice_divs.append(f"<div class='rewritten-choice'><b>{sym}</b></div>")
            else:
                choice_divs.append(
                    f"<div class='rewritten-choice'><b>{sym}</b> {render_english_inline_html(cleaned)}</div>"
                )
        choices_html = f"<div class='rewritten-choices-wrap'>{''.join(choice_divs)}</div>"

    return f"<div class='rewritten-card'>{''.join(body_parts)}{choices_html}</div>"


def _render_rewritten_card_html(rw: RewrittenQuestion) -> str:
    """개조 문항의 <자료>·<보기> 분리 박스, 화학식 첨자(영어/국어 전용 렌더러 분기), 선지 HTML을 모던 학습지 스타일로 조립한다."""
    if getattr(rw, "subject_domain", "") == "영어":
        return _render_english_rewritten_card_html(rw)
    from src.text_formatter import (
        format_chemical_scripts,
        render_korean_inline_html,
        split_stem_blocks,
    )

    is_kor = getattr(rw, "subject_domain", "") == "국어"
    fmt_body = (lambda s: render_korean_inline_html(s)) if is_kor else (lambda s: _html.escape(s).replace(chr(10), "<br/>"))

    blocks = split_stem_blocks(rw.rewritten_stem)
    body_parts: list[str] = []
    for kind, content in blocks:
        if kind == "data":
            if isinstance(content, dict):
                d_title = str(content.get("title") or "< 자 료 >")
                d_body = str(content.get("body") or "")
            else:
                d_title = "< 자 료 >"
                d_body = str(content)
            line_divs = "".join(
                "<div style='height:0.35rem;'></div>"
                if not ln.strip()
                else f"<div class='data-line'>{fmt_body(ln.strip())}</div>"
                for ln in d_body.split("\n")
            )
            body_parts.append(
                f"<div class='data-box'><div class='data-box-title'>{_html.escape(d_title)}</div><div>{line_divs}</div></div>"
            )
        elif kind == "bogi":
            bogi_rows = "".join(
                f"<div class='bogi-item'><span class='bogi-marker'>{_html.escape(mk)}</span>"
                f"<span>{fmt_body(txt)}</span></div>"
                for mk, txt in content
            )
            body_parts.append(
                f"<div class='bogi-box'><div class='bogi-title'>〈 보 기 〉</div>{bogi_rows}</div>"
            )
        elif kind == "bogi_prose":
            bogi_lines = "".join(
                "<div style='height:0.35rem;'></div>"
                if not ln.strip()
                else f"<div class='bogi-prose-line'>{fmt_body(ln.strip())}</div>"
                for ln in str(content).split("\n")
            )
            body_parts.append(
                f"<div class='bogi-box'><div class='bogi-title'>〈 보 기 〉</div><div class='bogi-prose'>{bogi_lines}</div></div>"
            )
        else:
            stem_divs = "".join(
                "<div style='height:0.35rem;'></div>"
                if not ln.strip()
                else f"<div class='stem-line'>{fmt_body(ln.strip())}</div>"
                for ln in str(content).split("\n")
            )
            body_parts.append(f"<div class='rewritten-stem'>{stem_divs}</div>")

    choices_html = ""
    if rw.rewritten_choices:
        choice_divs = []
        for ci, c_text in enumerate(rw.rewritten_choices):
            sym = CIRCLED_MAP.get(str(ci + 1), f"{ci + 1})")
            if is_kor:
                c_html = render_korean_inline_html(c_text.strip(), bold_to_underline=True)
            else:
                formatted_choice = format_chemical_scripts(c_text.strip())
                c_html = _html.escape(formatted_choice).replace("\n", "<br/>")
            choice_divs.append(f"<div class='rewritten-choice'><b>{sym}</b> {c_html}</div>")
        choices_html = f"<div class='rewritten-choices-wrap'>{''.join(choice_divs)}</div>"

    return f"<div class='rewritten-card'>{''.join(body_parts)}{choices_html}</div>"




render_english_rewritten_card_html = _render_english_rewritten_card_html
render_rewritten_card_html = _render_rewritten_card_html


def _gate_regen_model(ctx: AppContext, model_name: str) -> str:
    """재변환 직전 최신 사용자 한도를 확인해 상위 모델 한도 소진 시 3.5 Flash-Lite로 우회시킨다."""
    if not is_premium_model(model_name):
        return model_name
    fresh_user = get_user_by_username(ctx.storage, ctx.current_user.username) or ctx.current_user
    gated_model, gate_notice = resolve_model_for_user(fresh_user, model_name)
    if gate_notice:
        st.warning(gate_notice)
    return gated_model


def _charge_regen_usage(ctx: AppContext, model_name: str) -> None:
    """재변환에 상위 모델이 실제 사용되었다면 1문항 분량을 주간 한도/해금 크레딧에서 차감한다."""
    if is_premium_model(model_name):
        consume_premium_model_usage(ctx.storage, ctx.current_user.username, model_name, count=1)


gate_regen_model = _gate_regen_model
charge_regen_usage = _charge_regen_usage


def _record_english_seen_for_user(ctx: AppContext, items: list[dict[str, Any]]) -> None:
    """현재 계정이 PDF 생성 또는 풀이 채점을 완료한 영어 변형 문항을 기록해 중복 출제를 막는다."""
    if ctx.current_user is None or not items:
        return
    pairs: list[tuple[str, str]] = []
    for it in items:
        rw_item: RewrittenQuestion | None = it.get("rewritten")
        eng_meta = it.get("english_meta") or {}
        if rw_item is None or not eng_meta:
            continue
        passage = str(eng_meta.get("source_passage") or "").strip()
        t_type = str(eng_meta.get("target_type") or rw_item.target_question_type or "").strip()
        if not passage or not t_type:
            continue
        c_key = ctx.storage.build_english_pool_key(
            source_passage=passage,
            target_type=t_type,
            difficulty=str(eng_meta.get("difficulty") or ""),
            blank_unit=str(eng_meta.get("blank_unit") or ""),
            vocab_style=str(eng_meta.get("vocab_style") or ""),
        )
        pairs.append((c_key, str(rw_item.rewritten_stem or "")))
    if pairs:
        ctx.storage.mark_user_seen_english_variants(ctx.current_user.username, pairs)


record_english_seen_for_user = _record_english_seen_for_user


def _infer_exam_subject(items: list[dict[str, Any]]) -> str:
    from src.models import infer_subject_category

    subjects: list[str] = []
    for it in items:
        q_obj: QuestionRecord = it["question"]
        rw_obj: RewrittenQuestion | None = it.get("rewritten")
        if it.get("english_meta") or (rw_obj and getattr(rw_obj, "subject_domain", "") == "영어"):
            subj = "영어"
        else:
            subj = str(getattr(q_obj, "subject_category", "") or "").strip()
            if not subj:
                subj = infer_subject_category(getattr(q_obj, "curriculum_subject", "") or getattr(q_obj, "exam_name", ""))
        if subj and subj not in subjects:
            subjects.append(subj)
    if not subjects:
        return "기타"
    if len(subjects) == 1:
        return subjects[0]
    return "통합"


def _build_exam_sig(items: list[dict[str, Any]]) -> str:
    payload = [
        (
            it["question"].id,
            getattr(it.get("rewritten"), "rewritten_stem", ""),
            getattr(it.get("rewritten"), "rewritten_choices", []),
        )
        for it in items
    ]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _serialize_matched_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for it in items:
        q_obj = it.get("question")
        j_obj = it.get("judgment")
        rw_obj = it.get("rewritten")
        serialized.append(
            {
                "question": q_obj.model_dump() if hasattr(q_obj, "model_dump") else q_obj,
                "judgment": j_obj.model_dump() if hasattr(j_obj, "model_dump") else None,
                "rewritten": rw_obj.model_dump() if hasattr(rw_obj, "model_dump") else None,
                "concept_reference": str(it.get("concept_reference") or ""),
                "english_meta": dict(it.get("english_meta") or {}) if it.get("english_meta") else None,
                "korean_meta": dict(it.get("korean_meta") or {}) if it.get("korean_meta") else None,
            }
        )
    return serialized




def _deserialize_matched_items(raw_items: list[dict[str, Any]], storage: StorageManager) -> list[dict[str, Any]]:
    from src.models import QuestionMatchJudgment, QuestionRecord, RewrittenQuestion

    restored: list[dict[str, Any]] = []
    for raw in raw_items:
        try:
            q_obj = QuestionRecord.model_validate(raw["question"])
        except Exception:
            continue
        j_obj = None
        if raw.get("judgment"):
            try:
                j_obj = QuestionMatchJudgment.model_validate(raw["judgment"])
            except Exception:
                j_obj = None
        rw_obj = None
        if raw.get("rewritten"):
            try:
                rw_obj = RewrittenQuestion.model_validate(raw["rewritten"])
            except Exception:
                rw_obj = None
        crop_b = storage.get_crop_bytes(q_obj.crop_path) if getattr(q_obj, "crop_path", "") else None
        restored.append(
            {
                "question": q_obj,
                "judgment": j_obj,
                "crop_png_bytes": crop_b,
                "rewritten": rw_obj,
                "concept_reference": str(raw.get("concept_reference") or ""),
                "english_meta": raw.get("english_meta"),
                "korean_meta": raw.get("korean_meta"),
            }
        )
    return restored




deserialize_matched_items = _deserialize_matched_items
infer_exam_subject = _infer_exam_subject
build_exam_sig = _build_exam_sig
serialize_matched_items = _serialize_matched_items


def _auto_save_current_exam(
    ctx: AppContext,
    items: list[dict[str, Any]],
    title: str = "",
    pdf_bytes: bytes | None = None,
    score_info: dict[str, Any] | None = None,
    sig: str = "",
) -> None:
    if ctx.current_user is None or not items:
        return
    exam_id = (sig or _build_exam_sig(items))[:16]
    subject = _infer_exam_subject(items)
    record: dict[str, Any] = {
        "exam_id": exam_id,
        "title": (title or f"{subject} 맞춤형 시험지 ({len(items)}문항)").strip(),
        "subject": subject,
        "question_count": len(items),
        "items": _serialize_matched_items(items),
    }
    if score_info is not None:
        record["last_score"] = score_info
    ctx.storage.save_user_exam(ctx.current_user.username, record, pdf_bytes=pdf_bytes)


auto_save_current_exam = _auto_save_current_exam


def create_lazy_tabs(labels: Sequence[str], *, key: str) -> Any:
    """지원되는 Streamlit 버전에서 활성 탭만 지연 렌더링하도록 on_change='rerun'으로 탭을 생성한다."""
    try:
        return st.tabs(labels, on_change="rerun", key=key)
    except TypeError:
        return st.tabs(labels)


def create_lazy_expander(label: str, *, expanded: bool = False, key: str) -> Any:
    """지원되는 Streamlit 버전에서 펼쳐진 expander만 내부 위젯을 렌더링하도록 생성한다."""
    try:
        return st.expander(label, expanded=expanded, on_change="rerun", key=key)
    except TypeError:
        return st.expander(label, expanded=expanded)


def is_container_open(container: Any) -> bool:
    """탭/익스팬더가 현재 열려 있는지(혹은 open 속성이 없는 구버전인지) 판별한다."""
    is_open = getattr(container, "open", None)
    return True if is_open is None else bool(is_open)
