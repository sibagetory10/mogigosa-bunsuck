from __future__ import annotations

import math
import tempfile
import types as py_types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.config import FLASH_3_6_MODEL, FLASH_LITE_MODEL
from src.english_v3.assembler import ORDER_CHOICES, AssemblyError, assemble_question
from src.english_v3.batch_runner import submit_generation_batch, sync_and_advance_batch_job
from src.english_v3.discourse import rank_insertion_targets, rank_order_splits
from src.english_v3.pipeline import generate_batch_questions_v3, generate_english_transformed_question
from src.english_v3.preprocessor import (
    build_indexed_passage,
    check_type_feasibility,
    extract_passage_key_vocab,
    needs_passage_restoration,
    reconcile_exam_source_passage,
    split_english_sentences,
)
from src.english_v3.routing import (
    classify_target_family,
    get_type_routing_rule,
    resolve_execution_tier,
)
from src.english_v3.rules_gate import validate_assembled_question
from src.english_v3.schemas import (
    AbcVocabSlot,
    BlindSolverVerdict,
    ChoiceAbcSpec,
    GistSpec,
    InsertionSpec,
    IrrelevantSpec,
    OrderSpec,
    SpanInferenceSpec,
    SummaryPair,
    SummarySpec,
    Underline5Spec,
    UnderlineTargetItem,
)
from src.english_v3.transport import call_structured_v3, extract_json_object_text
from src.gemini.usage import bind_usage_tracker, calculate_token_cost_usd
from src.models.question import QuestionRecord
from src.pdf.exam_pdf import generate_custom_exam_pdf
from src.storage import StorageManager
from src.storage.english_pool_store import CURRENT_ENGLISH_POOL_VERSION, _is_valid_cached_english_question
from src.text_formatter import split_english_stem_blocks

SAMPLE_PASSAGE_6 = (
    "다음 글의 주제로 가장 적절한 것은?\n"
    "Dr. Smith notes that modern cities rely on distributed sensor networks in the U.S. today. "
    "These sensors collect real-time traffic and energy data across neighborhoods. "
    "However, raw measurements alone rarely change public policy. "
    "City planners must translate complex figures into clear narratives for citizens. "
    "When residents understand how local choices affect emissions, participation rises sharply. "
    "Therefore, transparent communication turns technical data into collective civic action.\n"
    "* emission: 배출"
)


def _sample_record(q_num: int = 23) -> QuestionRecord:
    return QuestionRecord(
        id=f"2026_고3_6월_영어__Q{q_num:02d}",
        exam_name="2026년 6월 고3 영어",
        q_num=q_num,
        subject_category="영어",
        curriculum_subject="영어",
        unit_name="주제 파악",
        min_grade=3,
        core_principles=["영어 독해"],
        required_terms=[],
        answer="1",
        official_explanation="해설",
        passage_text=SAMPLE_PASSAGE_6,
    )


def test_preprocessor_strips_noise_preserves_footnotes_and_abbreviations() -> None:
    indexed = build_indexed_passage(SAMPLE_PASSAGE_6)
    assert len(indexed.sentences) == 6
    assert "Dr. Smith" in indexed.sentences[0]
    assert "U.S. today." in indexed.sentences[0]
    assert indexed.footnotes == ("* emission: 배출",)
    assert "다음 글의 주제로" not in indexed.clean_text

    short_idx = build_indexed_passage("First sentence is here. Second sentence ends.")
    assert check_type_feasibility(short_idx, "order") is not None
    assert check_type_feasibility(short_idx, "insertion") is not None
    assert check_type_feasibility(indexed, "insertion") is None
    assert needs_passage_restoration("Short _____ blank.", "빈칸") is True
    assert needs_passage_restoration(indexed.clean_text, "주제") is False


def test_pool_version_and_routing_policy() -> None:
    assert CURRENT_ENGLISH_POOL_VERSION == "v3"
    assert resolve_execution_tier(2) == ("standard", False)
    assert resolve_execution_tier(3) == ("flex", False)
    assert resolve_execution_tier(5) == ("flex", False)
    assert classify_target_family("문맥상 어휘 적절성", "선택형 ((A)(B)(C) 네모)") == "choice_abc"
    assert classify_target_family("문맥상 어휘 적절성", "밑줄형 (①~⑤)") == "underline_5"

    r_gist = get_type_routing_rule("주제 파악", profile_name="cost_saver_lite")
    assert r_gist.generate.model == FLASH_LITE_MODEL
    assert r_gist.generate.thinking_level == "minimal"
    assert r_gist.require_blind_solver is False

    r_blank = get_type_routing_rule("빈칸 추론", profile_name="cost_saver_lite")
    assert r_blank.generate.thinking_level == "low"
    assert r_blank.retry_2_generate.thinking_level == "medium"
    assert r_blank.require_blind_solver is True
    assert r_blank.blind_solver is not None and r_blank.blind_solver.thinking_level == "low"

    r_vocab = get_type_routing_rule("문맥상 어휘 적절성", profile_name="cost_saver_lite")
    assert r_vocab.blind_solver is not None and r_vocab.blind_solver.thinking_level == "minimal"

    r_hybrid = get_type_routing_rule("어법상 틀린 것 (밑줄 ①~⑤)", profile_name="hybrid_flash_36")
    assert r_hybrid.retry_2_generate.model == FLASH_3_6_MODEL
    assert r_hybrid.blind_solver is not None and r_hybrid.blind_solver.model == FLASH_3_6_MODEL

    # Zero-token CSAT seed vocabulary extraction check
    kv = extract_passage_key_vocab(SAMPLE_PASSAGE_6)
    assert len(kv) >= 1 and all(":" in item for item in kv)


def test_assemble_all_11_types_and_render_pdf() -> None:
    indexed = build_indexed_passage(SAMPLE_PASSAGE_6)

    specs: list[tuple[str, str, str, object]] = [
        (
            "요지 파악",
            "구 (3~6단어)",
            "밑줄형 (①~⑤)",
            GistSpec(
                correct_choice="데이터의 투명한 공유가 시민의 환경 행동 참여를 이끈다.",
                distractors=[
                    "도시 센서 설치 비용이 매년 급증하고 있다.",
                    "기술적 수치는 전문가만 독점적으로 해석해야 한다.",
                    "교통 센서 네트워크는 전력 소비를 크게 줄여 준다.",
                    "모든 도시 정책은 시민 투표만으로 결정되어야 한다.",
                ],
                explanation_ko="데이터를 명확한 이야기로 전달해야 시민 참여가 높아진다는 내용이다.",
            ),
        ),
        (
            "주제 파악",
            "구 (3~6단어)",
            "밑줄형 (①~⑤)",
            GistSpec(
                correct_choice="role of clear communication in turning data into civic action",
                distractors=[
                    "limitations of sensor accuracy in rural neighborhoods",
                    "reasons why citizens ignore public energy policies",
                    "ways to lower hardware costs of traffic sensors",
                    "history of urban planning across major cities",
                ],
                explanation_ko="도시 데이터를 시민 행동으로 전환하는 소통의 역할을 다룬다.",
            ),
        ),
        (
            "제목 추론",
            "구 (3~6단어)",
            "밑줄형 (①~⑤)",
            GistSpec(
                correct_choice="Beyond Raw Numbers: How Storytelling Sparks Civic Action",
                distractors=[
                    "Why Sensors Fail Without Expensive Hardware",
                    "Keep Citizens Away from Complex Urban Data",
                    "Counting Traffic: A Guide for Sensor Engineers",
                    "The Future of Global Energy Production",
                ],
                explanation_ko="원시 데이터를 넘어 서사적 소통이 시민 행동을 이끈다는 제목이 적절하다.",
            ),
        ),
        (
            "함축적 의미 추론",
            "구 (3~6단어)",
            "밑줄형 (①~⑤)",
            SpanInferenceSpec(
                target_sentence_idx=3,
                exact_target_span="translate complex figures into clear narratives",
                correct_choice="present technical data in understandable ways for the public",
                distractors=[
                    "hide sensitive statistics from neighborhood residents",
                    "replace all human planners with automated sensors",
                    "collect more raw numbers before making any report",
                    "publish academic papers on urban sensor engineering",
                ],
                explanation_ko="복잡한 수치를 시민이 이해하기 쉬운 이야기로 풀어낸다는 의미다.",
            ),
        ),
        (
            "빈칸 추론",
            "구 (3~6단어)",
            "밑줄형 (①~⑤)",
            SpanInferenceSpec(
                target_sentence_idx=5,
                exact_target_span="turns technical data into collective civic action",
                correct_choice="transforms raw figures into shared public participation",
                distractors=[
                    "keeps local residents away from policy debates",
                    "reduces the maintenance cost of traffic networks",
                    "limits energy reporting to municipal officials",
                    "eliminates the need for real-time neighborhood sensors",
                ],
                explanation_ko="투명한 소통이 데이터를 시민 참여로 전환한다는 흐름이 적절하다.",
            ),
        ),
        (
            "어법상 틀린 것 (밑줄 ①~⑤)",
            "구 (3~6단어)",
            "밑줄형 (①~⑤)",
            Underline5Spec(
                items=[
                    UnderlineTargetItem(sentence_idx=0, exact_span="rely on", occurrence=1),
                    UnderlineTargetItem(sentence_idx=1, exact_span="collect", occurrence=1),
                    UnderlineTargetItem(sentence_idx=2, exact_span="change", occurrence=1),
                    UnderlineTargetItem(sentence_idx=4, exact_span="how", occurrence=1),
                    UnderlineTargetItem(sentence_idx=5, exact_span="turns", occurrence=1),
                ],
                wrong_item_index=2,
                mutated_span="changing",
                explanation_ko="③ 문장의 본동사 자리이므로 changing이 아니라 change가 적절하다.",
            ),
        ),
        (
            "문맥상 어휘 적절성",
            "구 (3~6단어)",
            "밑줄형 (①~⑤)",
            Underline5Spec(
                items=[
                    UnderlineTargetItem(sentence_idx=1, exact_span="collect", occurrence=1),
                    UnderlineTargetItem(sentence_idx=2, exact_span="rarely", occurrence=1),
                    UnderlineTargetItem(sentence_idx=3, exact_span="clear", occurrence=1),
                    UnderlineTargetItem(sentence_idx=4, exact_span="sharply", occurrence=1),
                    UnderlineTargetItem(sentence_idx=5, exact_span="transparent", occurrence=1),
                ],
                wrong_item_index=3,
                mutated_span="barely",
                explanation_ko="④ 시민이 이해하면 참여가 높아져야 하므로 barely는 문맥상 어색하다.",
            ),
        ),
        (
            "문맥상 어휘 적절성",
            "구 (3~6단어)",
            "선택형 ((A)(B)(C) 네모)",
            ChoiceAbcSpec(
                slots=[
                    AbcVocabSlot(sentence_idx=2, exact_span="rarely", occurrence=1, foil_span="automatically"),
                    AbcVocabSlot(sentence_idx=3, exact_span="clear", occurrence=1, foil_span="obscure"),
                    AbcVocabSlot(sentence_idx=5, exact_span="transparent", occurrence=1, foil_span="secretive"),
                ],
                explanation_ko="(A) rarely, (B) clear, (C) transparent가 글의 논지에 부합한다.",
            ),
        ),
        (
            "전체 흐름과 무관한 문장",
            "구 (3~6단어)",
            "밑줄형 (①~⑤)",
            IrrelevantSpec(
                insert_after_sentence_idx=2,
                intruder_sentence="Many sensor manufacturers also export waterproof batteries to coastal fishing vessels worldwide.",
                explanation_ko="③ 문장은 해안 어선에 대한 배터리 수출 이야기로 도시 데이터 소통과 무관하다.",
            ),
        ),
        (
            "글의 순서 ((A)-(B)-(C))",
            "구 (3~6단어)",
            "밑줄형 (①~⑤)",
            OrderSpec(
                first_chunk_start_idx=2,
                second_chunk_start_idx=3,
                third_chunk_start_idx=5,
                cohesion_rationale_ko="주어진 글의 센서 데이터 수집 뒤 (However) 한계 -> 기획자의 전환 -> 따라서 결론으로 이어진다.",
            ),
        ),
        (
            "주어진 문장 삽입",
            "구 (3~6단어)",
            "밑줄형 (①~⑤)",
            InsertionSpec(
                extracted_sentence_idx=2,
                cohesion_rationale_ko="However가 앞 문장의 데이터 수집과 뒤 문장의 서사 필요성을 연결한다.",
            ),
        ),
        (
            "요약문 완성 ((A), (B))",
            "구 (3~6단어)",
            "밑줄형 (①~⑤)",
            SummarySpec(
                summary_sentence="Urban sensor data can drive civic action only when planners (A) it into (B) narratives for residents.",
                correct_pair=SummaryPair(word_a="translate", word_b="accessible"),
                distractor_pairs=[
                    SummaryPair(word_a="conceal", word_b="accessible"),
                    SummaryPair(word_a="translate", word_b="cryptic"),
                    SummaryPair(word_a="abandon", word_b="technical"),
                    SummaryPair(word_a="restrict", word_b="costly"),
                ],
                explanation_ko="데이터를 시민이 접근 가능한 이야기로 번역해야 행동으로 이어진다.",
            ),
        ),
    ]

    pdf_items: list[dict] = []
    for idx_t, (t_type, b_unit, v_style, spec) in enumerate(specs, start=1):
        rw, meta = assemble_question(
            indexed,
            spec,
            target_type=t_type,
            blank_unit=b_unit,
            vocab_style=v_style,
            seed_val=100 + idx_t,
        )
        violations = validate_assembled_question(indexed, spec, rw, meta)
        assert violations == [], f"{t_type} ({v_style}) failed Gate 1: {violations}"
        assert _is_valid_cached_english_question(rw) is True
        blocks = split_english_stem_blocks(rw.rewritten_stem)
        assert blocks and blocks[0][0] == "instruction"
        if t_type == "글의 순서 ((A)-(B)-(C))":
            assert rw.rewritten_choices == list(ORDER_CHOICES)
        pdf_items.append({
            "question": _sample_record(idx_t),
            "judgment": None,
            "crop_png_bytes": None,
            "rewritten": rw,
            "concept_reference": f"📚 출처: 2026년 6월 고3 영어 {idx_t}번",
        })

    pdf_bytes = generate_custom_exam_pdf(
        items=pdf_items,
        exam_title="영어 v3 전 유형 통합 테스트",
        source_label="2026년 6월 고3 영어",
        two_column=True,
        include_original_for_rewritten=False,
    )
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 5000


def test_rules_gate_catches_source_mutation_blank_copy_and_length_bias() -> None:
    indexed = build_indexed_passage(SAMPLE_PASSAGE_6)

    # 1) Blank verbatim copy blocked
    copy_spec = SpanInferenceSpec(
        target_sentence_idx=5,
        exact_target_span="turns technical data into collective civic action",
        correct_choice="turns technical data into collective civic action",
        distractors=[
            "keeps citizens away from policy decisions",
            "reduces hardware costs for local sensors",
            "limits data access to municipal engineers",
            "replaces public narratives with raw tables",
        ],
        explanation_ko="원문 표현을 그대로 복사한 경우 테스트용 해설입니다.",
    )
    rw_copy, meta_copy = assemble_question(indexed, copy_spec, target_type="빈칸 추론", blank_unit="구 (3~6단어)")
    v_copy = validate_assembled_question(indexed, copy_spec, rw_copy, meta_copy)
    assert any("패러프레이징" in v for v in v_copy)

    # 2) Extreme length outlier blocked
    long_spec = GistSpec(
        correct_choice="the crucial role of transparent municipal communication in transforming complex technical sensor figures into meaningful collective civic participation",
        distractors=[
            "urban sensor hardware costs",
            "traffic data collection speed",
            "privacy risks of sensors",
            "history of city planning",
        ],
        explanation_ko="원문 표현을 그대로 복사한 경우 테스트용 해설입니다.",
    )
    rw_long, meta_long = assemble_question(indexed, long_spec, target_type="주제 파악")
    v_long = validate_assembled_question(indexed, long_spec, rw_long, meta_long)
    assert any("길이 편차" in v for v in v_long)

    # 3) Non-target sentence mutation blocked
    rw_mut = rw_long.model_copy(
        update={"rewritten_stem": rw_long.rewritten_stem.replace("neighborhoods.", "districts.")}
    )
    v_mut = validate_assembled_question(indexed, long_spec, rw_mut, meta_long)
    assert any("원문 불변성" in v for v in v_mut)


def test_flex_fallback_to_standard_and_50pct_discount_usage() -> None:
    std_cost = calculate_token_cost_usd(1_000_000, 0, 1_000_000, FLASH_LITE_MODEL, service_tier="standard")
    flex_cost = calculate_token_cost_usd(1_000_000, 0, 1_000_000, FLASH_LITE_MODEL, service_tier="flex")
    assert math.isclose(flex_cost, std_cost * 0.5, rel_tol=1e-9)

    fake_resp = py_types.SimpleNamespace(
        parsed=GistSpec(
            correct_choice="role of clear storytelling in civic engagement",
            distractors=[
                "sensor battery maintenance",
                "hiding municipal traffic data",
                "counting vehicles at night",
                "global economic trends today",
            ],
            explanation_ko="명확한 서사가 시민 참여를 이끈다는 설명입니다.",
        ),
        usage_metadata=py_types.SimpleNamespace(
            prompt_token_count=1000,
            cached_content_token_count=0,
            candidates_token_count=200,
            thoughts_token_count=0,
            total_token_count=1200,
        ),
    )

    mock_client = MagicMock()
    # First call (flex) raises 503 overloaded -> second call (standard fallback) succeeds
    mock_client.models.generate_content.side_effect = [
        RuntimeError("503 UNAVAILABLE: flex capacity overloaded"),
        fake_resp,
    ]
    recorded: list = []
    route = get_type_routing_rule("주제 파악").generate
    with bind_usage_tracker(recorded.append):
        res = call_structured_v3(
            mock_client,
            route=route,
            user_prompt="test",
            schema_cls=GistSpec,
            service_tier="flex",
        )
    assert isinstance(res, GistSpec)
    assert mock_client.models.generate_content.call_count == 2
    assert recorded[0].service_tier == "standard"


def test_pipeline_blind_solver_retry_escalation_and_batch(tmp_path: Path | None = None) -> None:
    with tempfile.TemporaryDirectory() as td:
        base_dir = Path(td) if tmp_path is None else tmp_path
        storage = StorageManager(local_root=base_dir / "exam_db", concept_root=base_dir / "concept_db")

    good_span_spec = SpanInferenceSpec(
        target_sentence_idx=5,
        exact_target_span="turns technical data into collective civic action",
        correct_choice="transforms raw figures into shared public participation",
        distractors=[
            "keeps local residents away from policy",
            "lowers hardware maintenance costs overall",
            "restricts reporting to senior engineers",
            "eliminates real-time neighborhood sensors",
        ],
        explanation_ko="투명한 소통이 데이터를 시민 참여로 바꾼다.",
    )

    seen_thinking_levels: list[str] = []

    def _fake_call(client, *, route, user_prompt, schema_cls, **kwargs):
        _ = (client, kwargs)
        if schema_cls is SpanInferenceSpec:
            seen_thinking_levels.append(route.thinking_level)
            return good_span_spec
        if schema_cls is BlindSolverVerdict:
            # Attempt 0: Gate 2 Blind Solver fails -> Attempt 1: escalated thinking ("medium") passes!
            if len(seen_thinking_levels) == 1:
                return BlindSolverVerdict(
                    selected_answer="2",
                    ambiguous_choices=["1"],
                    confidence="medium",
                    note_ko="2번 선지도 일부 방어 가능함",
                )
            lines = [ln for ln in user_prompt.splitlines() if "transforms raw figures" in ln]
            ans_char = "1"
            if lines:
                sym_map = {"①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5"}
                ans_char = sym_map.get(lines[0].strip()[:1], "1")
            return BlindSolverVerdict(
                selected_answer=ans_char,
                ambiguous_choices=[],
                confidence="high",
                note_ko="",
            )
        raise AssertionError(f"Unexpected schema {schema_cls}")

    with (
        patch("src.english_v3.pipeline.build_v3_client", return_value=MagicMock()),
        patch("src.english_v3.pipeline.call_structured_v3", side_effect=_fake_call),
        patch("src.english_v3.blind_validator.call_structured_v3", side_effect=_fake_call),
    ):
        rw = generate_english_transformed_question(
            api_key="dummy-key",
            model_name=FLASH_LITE_MODEL,
            source_passage=SAMPLE_PASSAGE_6,
            target_type="빈칸 추론",
            storage=storage,
            use_cache=True,
        )
        assert rw.subject_domain == "영어"
        assert seen_thinking_levels == ["low", "medium"]
        assert len(rw.replaced_terms_map) >= 1
        # Second call hits v3 cache pool without LLM calls
        seen_thinking_levels.clear()
        rw_cached = generate_english_transformed_question(
            api_key="dummy-key",
            model_name=FLASH_LITE_MODEL,
            source_passage=SAMPLE_PASSAGE_6,
            target_type="빈칸 추론",
            storage=storage,
            use_cache=True,
        )
        assert rw_cached.rewritten_stem == rw.rewritten_stem
        assert seen_thinking_levels == []


def test_async_batch_runner_2pass_flow(tmp_path: Path | None = None) -> None:
    with tempfile.TemporaryDirectory() as td:
        base_dir = Path(td) if tmp_path is None else tmp_path
        storage = StorageManager(local_root=base_dir / "exam_db", concept_root=base_dir / "concept_db")
    gist_spec = GistSpec(
        correct_choice="clear narratives driving civic action",
        distractors=[
            "rising costs of urban sensors",
            "secrecy in municipal energy reports",
            "sensor calibration methods in labs",
            "ancient history of public cities",
        ],
        explanation_ko="서사적 소통이 시민 행동을 이끈다.",
    )

    mock_client = MagicMock()
    mock_client.batches.create.side_effect = [
        py_types.SimpleNamespace(name="batches/p1_job", state="JOB_STATE_PENDING"),
        py_types.SimpleNamespace(name="batches/p2_job", state="JOB_STATE_PENDING"),
    ]

    with patch("src.english_v3.batch_runner.build_v3_client", return_value=mock_client):
        rec = submit_generation_batch(
            api_key="dummy",
            storage=storage,
            jobs=[
                {"passage_text": SAMPLE_PASSAGE_6, "target_type": "주제 파악"},
                {"passage_text": SAMPLE_PASSAGE_6, "target_type": "주어진 문장 삽입"},
            ],
        )
        assert rec["phase"] == "pass1_generate"
        assert rec["submitted_count"] == 2
        # Insertion is pre-assembled at 0 Pass-1 tokens; only Gist is sent to Pass 1 LLM!
        assert len(rec["manifest"]) == 1
        assert len(rec["preassembled_pass2"]) == 1
        pre_ans = rec["preassembled_pass2"][0]["rw_dump"]["preserved_answer"]

        # Pass 1 succeeds -> Gist committed immediately, pre-assembled Insertion enters Pass 2 Blind Solver
        p1_done = py_types.SimpleNamespace(
            state="JOB_STATE_SUCCEEDED",
            dest=py_types.SimpleNamespace(
                inlined_responses=[
                    py_types.SimpleNamespace(
                        metadata={"req_idx": "0"},
                        response=py_types.SimpleNamespace(text=gist_spec.model_dump_json()),
                    ),
                ]
            ),
        )
        p2_done = py_types.SimpleNamespace(
            state="JOB_STATE_SUCCEEDED",
            dest=py_types.SimpleNamespace(
                inlined_responses=[
                    py_types.SimpleNamespace(
                        metadata={"p2_idx": "0"},
                        response=py_types.SimpleNamespace(
                            text=BlindSolverVerdict(
                                selected_answer=pre_ans,
                                ambiguous_choices=[],
                                confidence="high",
                                note_ko="역접 연결어와 지시어가 앞뒤 문맥을 유일하게 이어 준다.",
                            ).model_dump_json()
                        ),
                    )
                ]
            ),
        )
        mock_client.batches.get.side_effect = [p1_done, p2_done]

        mid = sync_and_advance_batch_job("dummy", storage, rec["job_id"])
        assert mid["phase"] == "pass2_validate"
        assert mid["committed_count"] == 1
        assert len(mid["pending_pass2"]) == 1

        final = sync_and_advance_batch_job("dummy", storage, rec["job_id"])
        assert final["phase"] == "completed"
        assert final["committed_count"] == 2
        pool = storage.load_english_pool()
        assert len(pool) == 2


def test_order_empty_chunk_and_zero_gen_llm_fast_path(tmp_path: Path | None = None) -> None:
    with tempfile.TemporaryDirectory() as td:
        base_dir = Path(td) if tmp_path is None else tmp_path
        storage = StorageManager(local_root=base_dir / "exam_db", concept_root=base_dir / "concept_db")

        # Q30 regression: invalid OrderSpec where c1 == c2 (would create empty chunk) must fail assembler if direct
        indexed = build_indexed_passage(SAMPLE_PASSAGE_6)
        bad_order = OrderSpec(
            first_chunk_start_idx=2,
            second_chunk_start_idx=2,
            third_chunk_start_idx=4,
            cohesion_rationale_ko="충분히 긴 응집성 설명입니다.",
        )
        try:
            assemble_question(indexed, bad_order, target_type="글의 순서 ((A)-(B)-(C))")
            raise AssertionError("Expected AssemblyError for c1 == c2")
        except AssemblyError:
            pass

        passage_7 = (
            "Corporate wrongdoing has risen in recent decades. "
            "Internal employees often witness these illegal practices first. "
            "However, fear of retaliation frequently silences potential whistleblowers. "
            "To address this problem, lawmakers introduced legal protection programs. "
            "These safeguards shield informants from unfair dismissal. "
            "As a result, brave insiders have exposed several major scandals. "
            "Such disclosures ultimately strengthen public trust in markets."
        )

        called_schemas: list[type] = []

        def _fake_zero_gen_call(client, *, route, user_prompt, schema_cls, **kwargs):
            _ = (client, route, kwargs)
            called_schemas.append(schema_cls)
            assert schema_cls is BlindSolverVerdict, (
                f"Order/Insertion must NOT invoke Generator LLM, got {schema_cls}"
            )
            # Determine true answer from assembled prompt
            if "[주어진 문장]" in user_prompt:
                top_idx = rank_insertion_targets(build_indexed_passage(passage_7))[0][0]
                return BlindSolverVerdict(
                    selected_answer=str(top_idx),
                    ambiguous_choices=[],
                    confidence="high",
                    note_ko="역접 단서 However가 앞 문장의 목격과 뒤 문장의 침묵 원인을 연결한다.",
                )
            # For Order, inspect assembled chunks against chronological passage_7
            idx_7 = build_indexed_passage(passage_7)
            top_split = rank_order_splits(idx_7)[0][0]
            seed_check_rw, _ = assemble_question(
                idx_7,
                OrderSpec(
                    first_chunk_start_idx=top_split[0],
                    second_chunk_start_idx=top_split[1],
                    third_chunk_start_idx=top_split[2],
                ),
                target_type="글의 순서 ((A)-(B)-(C))",
                seed_val=int(__import__("hashlib").sha256(f"{idx_7.clean_text}||글의 순서 ((A)-(B)-(C))||0||".encode("utf-8")).hexdigest()[:8], 16),
            )
            return BlindSolverVerdict(
                selected_answer=seed_check_rw.preserved_answer,
                ambiguous_choices=[],
                confidence="high",
                note_ko="지시어 these problem과 These safeguards가 단락 순서를 유일하게 결정한다.",
            )

        with (
            patch("src.english_v3.pipeline.build_v3_client", return_value=MagicMock()),
            patch("src.english_v3.pipeline.call_structured_v3", side_effect=_fake_zero_gen_call),
            patch("src.english_v3.blind_validator.call_structured_v3", side_effect=_fake_zero_gen_call),
        ):
            rw_ins = generate_english_transformed_question(
                api_key="dummy",
                model_name=FLASH_LITE_MODEL,
                source_passage=passage_7,
                target_type="주어진 문장 삽입",
                storage=storage,
                use_cache=False,
            )
            rw_ord = generate_english_transformed_question(
                api_key="dummy",
                model_name=FLASH_LITE_MODEL,
                source_passage=passage_7,
                target_type="글의 순서 ((A)-(B)-(C))",
                storage=storage,
                use_cache=False,
            )
            # Total LLM calls for 2 questions (1 insertion + 1 order) == 2 BlindSolver calls ONLY (0 Generator calls!)
            assert called_schemas == [BlindSolverVerdict, BlindSolverVerdict]
            assert rw_ins.preserved_answer in ("1", "2", "3", "4", "5")
            assert "However" in rw_ins.pdf_grounded_explanation
            assert "safeguards" in rw_ord.pdf_grounded_explanation


def test_poulet_quote_preservation_and_source_reconciliation() -> None:
    # Q33 regression: closing quote " and closing paren ) must NEVER be stripped
    poulet = (
        'The very fact that all of our experiences "take place" in a physical space makes the space inseparable from what defines it. '
        "The tree and the ground beneath are inseparable from what took place there. "
        "This intertwining of space, place, and experience challenges the idea of place as passive backdrop. "
        "Spatial framework is the medium of all experiences — not merely a background exterior to experience (as commonly supposed). "
        'Belgian literary critic Georges Poulet wrote: "Beings surround themselves with the places where they find themselves. Without places, beings would be only abstractions."'
    )
    sents = split_english_sentences(poulet)
    assert any(s.endswith('(as commonly supposed).') for s in sents), f"Missing closing paren in {sents}"
    assert any(s.endswith('abstractions."') for s in sents), f"Missing closing quote in {sents}"

    idx_p = build_indexed_passage(poulet)
    ranked = rank_order_splits(idx_p)
    assert ranked
    top_triple, top_score, _ = ranked[0]
    # Top split must NOT cut inside Poulet's open quote (after sentence 4) and must prefer cohesive S2 boundary
    assert top_score > -50
    assert 2 in top_triple

    # Q36 regression: scrambled passage_text in DB reconciled deterministically from question_text + answer="4" ((C)-(A)-(B))
    q36_rec = QuestionRecord(
        id="2026_고1_10월_영어__Q36",
        exam_name="2026년 10월 고1 영어",
        q_num=36,
        subject_category="영어",
        curriculum_subject="영어",
        unit_name="글의 순서",
        min_grade=1,
        answer="4",
        official_explanation="해설",
        english_question_type="글의 순서",
        passage_text=(
            "Isaac Newton noted that light waves have no color until perceived by a brain. "
            "No, because sound requires a hearer to interpret pressure waves. "
            "That disturbance of molecules travels through the air silently until it hits an eardrum. "
            "The same is true of sound, which is merely molecular motion before perception. "
            "If a tree falls in an empty forest, does it make a sound?"
        ),
        question_text=(
            "36. 주어진 글 다음에 이어질 글의 순서로 가장 적절한 것은?\n"
            "Isaac Newton noted that light waves have no color until perceived by a brain.\n"
            "(A) No, because sound requires a hearer to interpret pressure waves. "
            "That disturbance of molecules travels through the air silently until it hits an eardrum.\n"
            "(B) If a tree falls in an empty forest, does it make a sound?\n"
            "(C) The same is true of sound, which is merely molecular motion before perception.\n"
            "① (A)-(C)-(B) ② (B)-(A)-(C) ③ (B)-(C)-(A) ④ (C)-(A)-(B) ⑤ (C)-(B)-(A)"
        ),
    )
    reconciled_q36 = reconcile_exam_source_passage(q36_rec.passage_text, q36_rec)
    # Chronological order (C)-(A)-(B): "The same is true of sound..." -> "No, because..." -> "If a tree falls..."
    assert reconciled_q36.index("The same is true of sound") < reconciled_q36.index("No, because")
    assert reconciled_q36.index("No, because") < reconciled_q36.index("If a tree falls")


def test_live_server_regression_schemas_json_fence_blank_root_and_long_markers() -> None:
    from google import genai
    from google.genai import _transformers, batches
    from src.english_v3.routing import StageRoute
    from src.english_v3.transport import _parse_v3_structured, build_inlined_request_dict

    # 1) All 9 Pydantic schemas must transform cleanly via google-genai SDK t_schema & Batch wire serializer
    dummy_client = genai.Client(api_key="dummy-test-key")
    all_specs = (
        GistSpec,
        SpanInferenceSpec,
        Underline5Spec,
        ChoiceAbcSpec,
        IrrelevantSpec,
        OrderSpec,
        InsertionSpec,
        SummarySpec,
        BlindSolverVerdict,
    )
    for cls in all_specs:
        _transformers.t_schema(dummy_client, cls)
        inlined = build_inlined_request_dict(
            route=StageRoute(),
            user_prompt="test",
            schema_cls=cls,
            metadata={"i": "0"},
        )
        batches._InlinedRequest_to_mldev(dummy_client._api_client, inlined)

    # 2) Conversational preamble + markdown fence JSON extraction
    fenced = (
        "Here is the JSON requested:\n```json\n"
        '{"selected_answer": "3", "ambiguous_choices": [], "confidence": "high", "note_ko": "테스트 근거입니다."}\n'
        "```"
    )
    extracted = extract_json_object_text(fenced)
    assert extracted.startswith("{") and extracted.endswith("}")
    parsed = _parse_v3_structured(py_types.SimpleNamespace(parsed=None, text=fenced), BlindSolverVerdict)
    assert parsed.selected_answer == "3"

    # 3) Blank word-unit paraphrasing: allow reflexive pronouns & distinct words sharing 5-char prefix; block same root
    indexed = build_indexed_passage(
        "People often deceive themselves when they try to adapt quickly. "
        "They rely on familiar habits across daily routines. "
        "However, clear feedback helps them adjust their goals effectively."
    )
    ok_reflexive_spec = SpanInferenceSpec(
        target_sentence_idx=0,
        exact_target_span="deceive themselves",
        correct_choice="mislead themselves",
        distractors=["help others", "ignore rumors", "praise friends", "avoid risks"],
        explanation_ko="재귀대명사 themselves는 기능어이므로 동사 패러프레이징 시 허용된다.",
    )
    rw_ok, meta_ok = assemble_question(indexed, ok_reflexive_spec, target_type="빈칸 추론", blank_unit="단어 (1~2단어)")
    assert validate_assembled_question(indexed, ok_reflexive_spec, rw_ok, meta_ok) == []

    bad_root_spec = SpanInferenceSpec(
        target_sentence_idx=0,
        exact_target_span="adapt quickly",
        correct_choice="adapting rapidly",
        distractors=["resting quietly", "moving slowly", "leaving early", "waiting calmly"],
        explanation_ko="동일 어근 adapt/adapting 반복은 차단되어야 한다.",
    )
    rw_bad, meta_bad = assemble_question(indexed, bad_root_spec, target_type="빈칸 추론", blank_unit="단어 (1~2단어)")
    assert any("패러프레이징" in v for v in validate_assembled_question(indexed, bad_root_spec, rw_bad, meta_bad))

    # 4) Long-passage (a)~(e) markers stripped cleanly
    long_raw = (
        "Researchers (a) observed that urban trees (b) reduce heat islands. "
        "They (c) measured street temperatures and (d) compared shade levels. "
        "Finally, city planners (e) expanded green corridors."
    )
    idx_long = build_indexed_passage(long_raw)
    assert not any(f"({ch})" in idx_long.clean_text for ch in ("a", "b", "c", "d", "e"))
    assert needs_passage_restoration(long_raw, "41~42번 장문 어휘") is True


class EnglishV3EngineTestCase(unittest.TestCase):
    def test_all(self) -> None:
        test_preprocessor_strips_noise_preserves_footnotes_and_abbreviations()
        test_pool_version_and_routing_policy()
        test_assemble_all_11_types_and_render_pdf()
        test_rules_gate_catches_source_mutation_blank_copy_and_length_bias()
        test_flex_fallback_to_standard_and_50pct_discount_usage()
        test_pipeline_blind_solver_retry_escalation_and_batch()
        test_async_batch_runner_2pass_flow()
        test_order_empty_chunk_and_zero_gen_llm_fast_path()
        test_poulet_quote_preservation_and_source_reconciliation()
        test_live_server_regression_schemas_json_fence_blank_root_and_long_markers()


if __name__ == "__main__":
    unittest.main()
