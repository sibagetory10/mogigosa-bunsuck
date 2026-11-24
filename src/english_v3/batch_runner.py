"""Asynchronous Gemini Batch API (50% discount) 2-Pass bulk pool builder for English Engine v3."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from google.genai import types

from src.config import FLASH_LITE_MODEL
from src.english_v3.assembler import AssemblyError, assemble_question
from src.english_v3.blind_validator import build_blind_solver_prompt, evaluate_blind_verdict
from src.english_v3.pipeline import build_generation_prompt, resolve_schema_for_type
from src.english_v3.preprocessor import build_indexed_passage, check_type_feasibility
from src.english_v3.routing import (
    classify_target_family,
    get_type_routing_rule,
    resolve_active_routing_profile,
)
from src.english_v3.rules_gate import validate_assembled_question
from src.english_v3.schemas import BlindSolverVerdict
from src.english_v3.transport import build_inlined_request_dict, build_v3_client
from src.models.concept import RewrittenQuestion
from src.storage.base import _atomic_write_text
from src.storage.english_pool_store import EnglishPoolStoreMixin

_TERMINAL_STATES = frozenset({
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
})


def _batch_jobs_path(storage: Any) -> Path:
    return Path(storage.local_root) / "english_batch_jobs.json"


def load_batch_jobs(storage: Any) -> list[dict[str, Any]]:
    """Thread-safely load tracked English Batch API jobs."""
    path = _batch_jobs_path(storage)
    with storage._lock:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return [dict(x) for x in data if isinstance(x, dict)] if isinstance(data, list) else []
        except Exception:
            return []


def _save_batch_jobs(storage: Any, jobs: list[dict[str, Any]]) -> None:
    path = _batch_jobs_path(storage)
    with storage._lock:
        _atomic_write_text(path, json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_inlined_response_text(inlined_resp: Any) -> str:
    resp = getattr(inlined_resp, "response", None)
    if resp is None and isinstance(inlined_resp, dict):
        resp = inlined_resp.get("response")
    if resp is None:
        return ""
    txt = getattr(resp, "text", None)
    if txt:
        return str(txt)
    candidates = getattr(resp, "candidates", None) or (resp.get("candidates") if isinstance(resp, dict) else None)
    if candidates:
        first = candidates[0]
        content = getattr(first, "content", None) or (first.get("content") if isinstance(first, dict) else None)
        parts = getattr(content, "parts", None) or (content.get("parts") if isinstance(content, dict) else None)
        if parts:
            return str(getattr(parts[0], "text", None) or (parts[0].get("text") if isinstance(parts[0], dict) else "") or "")
    return ""


def _extract_inlined_metadata(inlined_resp: Any) -> dict[str, Any]:
    meta = getattr(inlined_resp, "metadata", None)
    if meta is None and isinstance(inlined_resp, dict):
        meta = inlined_resp.get("metadata")
    return dict(meta) if isinstance(meta, dict) else {}


def submit_generation_batch(
    api_key: str,
    storage: Any,
    jobs: list[dict[str, Any]],
    *,
    display_label: str = "",
    routing_profile: str | None = None,
    max_per_key: int = 5,
) -> dict[str, Any]:
    """Submit Pass 1 (generation) requests to Gemini Batch API (50% discount)."""
    eff_profile = resolve_active_routing_profile(routing_profile)
    allow_escalation = eff_profile != "cost_saver_lite"
    pool = storage.load_english_pool() if hasattr(storage, "load_english_pool") else {}

    inlined_requests: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    skipped_cached = 0
    skipped_infeasible = 0

    for idx, job in enumerate(jobs):
        raw_p = str(job.get("passage_text") or "").strip()
        target_type = str(job.get("target_type") or "").strip()
        difficulty = str(job.get("difficulty") or "고3 평가원·수능 수준")
        blank_unit = str(job.get("blank_unit") or "구 (3~6단어)")
        vocab_style = str(job.get("vocab_style") or "밑줄형 (①~⑤)")
        indexed = build_indexed_passage(raw_p)
        family = classify_target_family(target_type, vocab_style=vocab_style)

        if check_type_feasibility(indexed, family):
            skipped_infeasible += 1
            continue

        ck = EnglishPoolStoreMixin.build_english_pool_key(
            source_passage=indexed.clean_text,
            target_type=target_type,
            difficulty=difficulty,
            blank_unit=blank_unit,
            vocab_style=vocab_style,
        )
        if len(pool.get(ck) or []) >= max_per_key:
            skipped_cached += 1
            continue

        rule = get_type_routing_rule(target_type, vocab_style=vocab_style, profile_name=eff_profile)
        schema_cls = resolve_schema_for_type(target_type, vocab_style=vocab_style)
        prompt = build_generation_prompt(
            indexed,
            target_type=target_type,
            difficulty=difficulty,
            blank_unit=blank_unit,
            vocab_style=vocab_style,
        )
        req_idx = len(manifest)
        inlined_requests.append(
            build_inlined_request_dict(
                route=rule.generate,
                user_prompt=prompt,
                schema_cls=schema_cls,
                metadata={"req_idx": str(req_idx)},
                allow_flash_escalation=allow_escalation,
            )
        )
        manifest.append({
            "req_idx": req_idx,
            "cache_key": ck,
            "clean_text": indexed.clean_text,
            "sentences": list(indexed.sentences),
            "footnotes": list(indexed.footnotes),
            "target_type": target_type,
            "difficulty": difficulty,
            "blank_unit": blank_unit,
            "vocab_style": vocab_style,
            "source_label": str(job.get("source_label") or ""),
        })

    if not inlined_requests:
        raise ValueError(
            f"제출할 신규 작업이 없습니다 (이미 캐시 충족: {skipped_cached}건, 구조 부적합: {skipped_infeasible}건)."
        )

    client = build_v3_client(api_key)
    job_id = uuid.uuid4().hex[:12]
    batch_job = client.batches.create(
        model=FLASH_LITE_MODEL,
        src=inlined_requests,
        config=types.CreateBatchJobConfig(
            display_name=display_label or f"mogigosa-eng-v3-p1-{job_id}",
        ),
    )
    record: dict[str, Any] = {
        "job_id": job_id,
        "display_label": display_label or f"영어 v3 배치 ({len(manifest)}건)",
        "routing_profile": eff_profile,
        "phase": "pass1_generate",
        "state": str(getattr(batch_job, "state", "JOB_STATE_PENDING")),
        "pass1_batch_name": str(getattr(batch_job, "name", "")),
        "pass2_batch_name": "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "submitted_count": len(manifest),
        "skipped_cached": skipped_cached,
        "skipped_infeasible": skipped_infeasible,
        "committed_count": 0,
        "rejected_count": 0,
        "manifest": manifest,
        "pending_pass2": [],
    }
    all_jobs = load_batch_jobs(storage)
    all_jobs.insert(0, record)
    _save_batch_jobs(storage, all_jobs[:50])
    return record


def sync_and_advance_batch_job(
    api_key: str,
    storage: Any,
    job_id: str,
) -> dict[str, Any]:
    """Poll Gemini Batch API status, run local Gate 1, submit Pass 2 Blind Solver, and commit passing items."""
    all_jobs = load_batch_jobs(storage)
    target_idx = next((i for i, j in enumerate(all_jobs) if j.get("job_id") == job_id), -1)
    if target_idx < 0:
        raise ValueError(f"배치 작업을 찾을 수 없습니다: {job_id}")

    rec = dict(all_jobs[target_idx])
    phase = str(rec.get("phase") or "")
    if phase in ("completed", "failed"):
        return rec

    client = build_v3_client(api_key)
    eff_profile = str(rec.get("routing_profile") or "cost_saver_lite")
    allow_escalation = eff_profile != "cost_saver_lite"

    if phase == "pass1_generate":
        b_job = client.batches.get(name=rec["pass1_batch_name"])
        b_state = str(getattr(b_job, "state", ""))
        rec["state"] = b_state
        rec["updated_at"] = datetime.now().isoformat(timespec="seconds")

        if b_state == "JOB_STATE_SUCCEEDED":
            dest = getattr(b_job, "dest", None)
            responses = list(getattr(dest, "inlined_responses", None) or [])
            manifest_map = {int(m["req_idx"]): m for m in rec.get("manifest") or []}
            pass2_requests: list[dict[str, Any]] = []
            pending_pass2: list[dict[str, Any]] = []
            committed = int(rec.get("committed_count") or 0)
            rejected = int(rec.get("rejected_count") or 0)

            from src.english_v3.preprocessor import IndexedPassage

            for i_resp, resp_item in enumerate(responses):
                meta = _extract_inlined_metadata(resp_item)
                req_idx = int(meta.get("req_idx", i_resp))
                m_entry = manifest_map.get(req_idx)
                if not m_entry:
                    continue
                raw_json = _extract_inlined_response_text(resp_item)
                if not raw_json.strip():
                    rejected += 1
                    continue

                t_type = m_entry["target_type"]
                v_style = m_entry.get("vocab_style", "밑줄형 (①~⑤)")
                b_unit = m_entry.get("blank_unit", "구 (3~6단어)")
                schema_cls = resolve_schema_for_type(t_type, vocab_style=v_style)
                rule = get_type_routing_rule(t_type, vocab_style=v_style, profile_name=eff_profile)

                try:
                    spec = schema_cls.model_validate_json(raw_json)
                    indexed = IndexedPassage(
                        clean_text=m_entry["clean_text"],
                        sentences=tuple(m_entry["sentences"]),
                        footnotes=tuple(m_entry.get("footnotes") or []),
                    )
                    rw, a_meta = assemble_question(
                        indexed,
                        spec,
                        target_type=t_type,
                        blank_unit=b_unit,
                        vocab_style=v_style,
                        seed_val=req_idx + 101,
                        self_note=f"v3 Batch Pass1 ({eff_profile})",
                    )
                    violations = validate_assembled_question(indexed, spec, rw, a_meta)
                    if violations:
                        rejected += 1
                        continue
                except (AssemblyError, Exception):
                    rejected += 1
                    continue

                if not rule.require_blind_solver or rule.blind_solver is None:
                    storage.save_cached_english_question(m_entry["cache_key"], rw)
                    committed += 1
                else:
                    p2_idx = len(pending_pass2)
                    pass2_requests.append(
                        build_inlined_request_dict(
                            route=rule.blind_solver,
                            user_prompt=build_blind_solver_prompt(rw),
                            schema_cls=BlindSolverVerdict,
                            metadata={"p2_idx": str(p2_idx)},
                            allow_flash_escalation=allow_escalation,
                        )
                    )
                    pending_pass2.append({
                        "p2_idx": p2_idx,
                        "cache_key": m_entry["cache_key"],
                        "rw_dump": rw.model_dump(),
                    })

            rec["committed_count"] = committed
            rec["rejected_count"] = rejected
            rec["pending_pass2"] = pending_pass2

            if pass2_requests:
                p2_batch = client.batches.create(
                    model=FLASH_LITE_MODEL,
                    src=pass2_requests,
                    config=types.CreateBatchJobConfig(
                        display_name=f"mogigosa-eng-v3-p2-{job_id}",
                    ),
                )
                rec["phase"] = "pass2_validate"
                rec["pass2_batch_name"] = str(getattr(p2_batch, "name", ""))
                rec["state"] = str(getattr(p2_batch, "state", "JOB_STATE_PENDING"))
            else:
                rec["phase"] = "completed"
        elif b_state in _TERMINAL_STATES:
            rec["phase"] = "failed"

    elif phase == "pass2_validate":
        b_job2 = client.batches.get(name=rec["pass2_batch_name"])
        b_state2 = str(getattr(b_job2, "state", ""))
        rec["state"] = b_state2
        rec["updated_at"] = datetime.now().isoformat(timespec="seconds")

        if b_state2 == "JOB_STATE_SUCCEEDED":
            dest2 = getattr(b_job2, "dest", None)
            responses2 = list(getattr(dest2, "inlined_responses", None) or [])
            p2_map = {int(p["p2_idx"]): p for p in rec.get("pending_pass2") or []}
            committed = int(rec.get("committed_count") or 0)
            rejected = int(rec.get("rejected_count") or 0)

            for i_r2, resp2 in enumerate(responses2):
                meta2 = _extract_inlined_metadata(resp2)
                p2_idx = int(meta2.get("p2_idx", i_r2))
                p2_entry = p2_map.get(p2_idx)
                if not p2_entry:
                    continue
                raw_v = _extract_inlined_response_text(resp2)
                if not raw_v.strip():
                    rejected += 1
                    continue
                try:
                    verdict = BlindSolverVerdict.model_validate_json(raw_v)
                    rw_obj = RewrittenQuestion.model_validate(p2_entry["rw_dump"])
                    ok, _ = evaluate_blind_verdict(verdict, rw_obj.preserved_answer)
                    if ok:
                        storage.save_cached_english_question(p2_entry["cache_key"], rw_obj)
                        committed += 1
                    else:
                        rejected += 1
                except Exception:
                    rejected += 1

            rec["committed_count"] = committed
            rec["rejected_count"] = rejected
            rec["pending_pass2"] = []
            rec["phase"] = "completed"
        elif b_state2 in _TERMINAL_STATES:
            rec["phase"] = "failed"

    all_jobs[target_idx] = rec
    _save_batch_jobs(storage, all_jobs)
    return rec
