"""Gate 2: Token-lean Answer-hidden Blind Solver validator."""

from __future__ import annotations

import re

from google import genai

from src.english_v3.routing import ExecutionTier, StageRoute
from src.english_v3.schemas import BlindSolverVerdict
from src.english_v3.transport import COMPACT_SOLVER_SYSTEM_PROMPT, call_structured_v3
from src.models.concept import RewrittenQuestion

CIRCLED_NUMS: tuple[str, ...] = ("①", "②", "③", "④", "⑤")
_FOOTNOTE_LINE_RE = re.compile(r"^\*{1,2}\s*[A-Za-z][A-Za-z\s\-']*:\s*.+$", re.MULTILINE)


def _trim_stem_for_solver(stem: str) -> str:
    cleaned = _FOOTNOTE_LINE_RE.sub("", stem or "")
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def build_blind_solver_prompt(rw: RewrittenQuestion, *, need_rationale_ko: bool = False) -> str:
    """Build a compact answer-hidden student-facing prompt for Blind Solver verification."""
    choices_lines = "\n".join(
        f"{CIRCLED_NUMS[i]} {c}" for i, c in enumerate(rw.rewritten_choices[:5])
    )
    t = (rw.target_question_type or "").strip()
    rule_hint = ""
    if "순서" in t:
        rule_hint = (
            "Check boundary anaphora (this/these/such/the+noun, 'this last part', 'that disturbance', "
            "'?'->'No/Yes...', however/instead/therefore) linking Given->1st->2nd->3rd.\n"
        )
    elif "삽입" in t:
        rule_hint = (
            "Match the backward referent/connector in [주어진 문장] with the exact gap ①~⑤ where a break occurs.\n"
        )

    note_rule = (
        "In `note_ko`, write 1 concise Korean sentence (<110 chars) explaining the cohesive clues for selected_answer (or defect if ambiguous)."
        if need_rationale_ko
        else "Set `note_ko`='' if confident single answer; write 1 short Korean reason ONLY if ambiguous or defective."
    )
    trimmed_stem = _trim_stem_for_solver(rw.rewritten_stem)
    return (
        f"{rule_hint}{note_rule}\n\n"
        f"{trimmed_stem}\n\n"
        f"{choices_lines}"
    )


def evaluate_blind_verdict(verdict: BlindSolverVerdict, intended_answer: str) -> tuple[bool, str]:
    """Evaluate BlindSolverVerdict against intended_answer with zero ambiguity tolerance."""
    intended = str(intended_answer or "").strip()
    selected = str(verdict.selected_answer or "").strip()
    other_ambig = [c for c in (verdict.ambiguous_choices or []) if c != intended]
    note = (verdict.note_ko or "").strip()

    if selected != intended:
        return (
            False,
            f"Blind Solver 불일치: 의도 정답 {intended}번 vs 검수 선택 {selected}번" + (f" ({note})" if note else ""),
        )
    if other_ambig:
        return (
            False,
            f"복수 정답 의심: 경합 선지 {','.join(other_ambig)}번" + (f" ({note})" if note else ""),
        )
    if verdict.confidence == "low":
        return (
            False,
            "단일 정답 확신도 미달(low)" + (f": {note}" if note else ""),
        )
    return True, note


def run_blind_solver_gate(
    client: genai.Client,
    *,
    rw: RewrittenQuestion,
    route: StageRoute,
    service_tier: ExecutionTier = "standard",
    cached_content_name: str | None = None,
    allow_flash_escalation: bool = False,
    need_rationale_ko: bool = False,
) -> tuple[bool, str]:
    """Run Gate 2 Blind Solver LLM verification; returns (passed, rationale_or_error)."""
    prompt = build_blind_solver_prompt(rw, need_rationale_ko=need_rationale_ko)
    verdict = call_structured_v3(
        client,
        route=route,
        user_prompt=prompt,
        schema_cls=BlindSolverVerdict,
        service_tier=service_tier,
        cached_content_name=cached_content_name,
        allow_flash_escalation=allow_flash_escalation,
        system_instruction_override=COMPACT_SOLVER_SYSTEM_PROMPT,
    )
    return evaluate_blind_verdict(verdict, rw.preserved_answer)
