"""Gate 2: Answer-hidden Blind Solver validator for logic/grammar-sensitive CSAT types."""

from __future__ import annotations

from google import genai

from src.english_v3.routing import ExecutionTier, StageRoute
from src.english_v3.schemas import BlindSolverVerdict
from src.english_v3.transport import call_structured_v3
from src.models.concept import RewrittenQuestion

CIRCLED_NUMS: tuple[str, ...] = ("①", "②", "③", "④", "⑤")


def build_blind_solver_prompt(rw: RewrittenQuestion) -> str:
    """Build an answer-hidden student-facing exam prompt for independent verification."""
    choices_lines = "\n".join(
        f"{CIRCLED_NUMS[i]} {c}" for i, c in enumerate(rw.rewritten_choices[:5])
    )
    t = (rw.target_question_type or "").strip()
    discourse_hint = ""
    if "순서" in t:
        discourse_hint = (
            "\nCRITICAL ORDERING RULE:\n"
            "- Check exact boundary-level anaphora linking END of [주어진 글] -> START of 1st block, "
            "END of 1st -> START of 2nd, and END of 2nd -> START of 3rd.\n"
            "- Prioritize explicit grammatical/lexical referents (this/these/such/the same + noun, "
            "'this last part', 'that disturbance', direct '?' -> 'No/Yes, because...', "
            "however/instead/therefore, definite 'the + noun') over loose thematic ordering.\n"
        )
    elif "삽입" in t:
        discourse_hint = (
            "\nCRITICAL INSERTION RULE:\n"
            "- Identify the exact anaphoric/contrastive hook inside [주어진 문장], and locate the gap ①~⑤ "
            "where BOTH its backward referent is satisfied AND a logical/referential break occurs without it.\n"
        )
    return (
        "You are an independent Korean CSAT (수능) English test-taker and KICE quality auditor.\n"
        "Solve the question below strictly based on the given passage, logic, cohesion, and grammar.\n"
        "Do NOT assume any pre-selected answer. Check whether EXACTLY ONE choice is uniquely correct.\n"
        "If another choice can also be reasonably defended, list it in `ambiguous_choices` and explain why in `defect_reason`.\n"
        f"{discourse_hint}\n"
        f"[Question Type]: {rw.target_question_type}\n\n"
        f"{rw.rewritten_stem}\n\n"
        f"[Choices]\n{choices_lines}\n"
    )


def evaluate_blind_verdict(verdict: BlindSolverVerdict, intended_answer: str) -> tuple[bool, str]:
    """Evaluate BlindSolverVerdict against intended_answer with zero ambiguity tolerance."""
    intended = str(intended_answer or "").strip()
    selected = str(verdict.selected_answer or "").strip()
    other_ambig = [c for c in (verdict.ambiguous_choices or []) if c != intended]

    if selected != intended:
        reason = verdict.defect_reason or verdict.brief_analysis
        return (
            False,
            f"Blind Solver 불일치: 의도된 정답은 {intended}번이나 검수 모델이 {selected}번을 선택함 ({reason})",
        )
    if other_ambig:
        reason = verdict.defect_reason or verdict.brief_analysis
        return (
            False,
            f"복수 정답 의심: {intended}번 외 경합 선지 {','.join(other_ambig)}번이 존재함 ({reason})",
        )
    if verdict.confidence == "low":
        reason = verdict.defect_reason or verdict.brief_analysis
        return (
            False,
            f"단일 정답 확신도 미달(low): {reason}",
        )
    return True, ""


def run_blind_solver_gate(
    client: genai.Client,
    *,
    rw: RewrittenQuestion,
    route: StageRoute,
    service_tier: ExecutionTier = "standard",
    cached_content_name: str | None = None,
    allow_flash_escalation: bool = False,
) -> tuple[bool, str]:
    """Run Gate 2 Blind Solver LLM verification on an assembled question."""
    prompt = build_blind_solver_prompt(rw)
    verdict = call_structured_v3(
        client,
        route=route,
        user_prompt=prompt,
        schema_cls=BlindSolverVerdict,
        service_tier=service_tier,
        cached_content_name=cached_content_name,
        allow_flash_escalation=allow_flash_escalation,
    )
    return evaluate_blind_verdict(verdict, rw.preserved_answer)
