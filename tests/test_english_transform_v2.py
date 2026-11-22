from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.gemini import (
    _assemble_rewritten_question_from_spec,
    _prepare_indexed_sentences_for_transform,
    _validate_english_transformed_output,
    generate_english_transformed_question,
    prepare_clean_english_source_passage,
)
from src.models import (
    EnglishGistTransformSpec,
    EnglishInsertionTransformSpec,
    EnglishIrrelevantTransformSpec,
    EnglishOrderTransformSpec,
    EnglishSpanTargetTransformSpec,
    EnglishSummaryPairItem,
    EnglishSummaryTransformSpec,
    EnglishUnderlineFiveItem,
    EnglishUnderlineFiveTransformSpec,
    EnglishVocabChoiceTransformSpec,
    EnglishVocabChoiceTripleItem,
)
from src.storage import StorageManager

SAMPLE_6_SENT_PASSAGE = (
    "Cognitive offloading refers to using physical action to alter the information processing requirements of a task so as to reduce cognitive demand. "
    "Instead of remembering a list of groceries inside our heads, we write the items down on a piece of paper or type them into a smartphone. "
    "This externalization frees up our limited working memory capacity for more complex analytical reasoning and creative problem solving. "
    "However, excessive reliance on external digital devices can gradually weaken our intrinsic episodic memory retrieval pathways over time. "
    "When individuals expect that information will remain continuously accessible online, they invest significantly less effort in encoding the details deeply. "
    "Therefore, educators emphasize balancing technological convenience with deliberate mental practice to maintain long-term cognitive resilience."
)


class TestEnglishTransformV2(unittest.TestCase):
    def test_source_sanitizer_reorders_and_cleans_exam_artifacts(self) -> None:
        raw_order_q = (
            "36. 주어진 글 다음에 이어질 글의 순서로 가장 적절한 것은?\n"
            "Cognitive offloading refers to using physical action to reduce cognitive demand.\n"
            "(A) This externalization frees up our limited working memory capacity for complex reasoning.\n"
            "(B) Instead of remembering a list of groceries in our heads, we write the items down on paper.\n"
            "(C) Therefore, educators emphasize balancing technological convenience with mental practice.\n"
            "① (A)-(C)-(B) ② (B)-(A)-(C) ③ (B)-(C)-(A) ④ (C)-(A)-(B) ⑤ (C)-(B)-(A)"
        )
        cleaned_order = prepare_clean_english_source_passage(
            source_passage="",
            question_text=raw_order_q,
            q_num=36,
            orig_question_type="글의 순서",
            orig_answer="2",
        )
        self.assertNotIn("(A)", cleaned_order)
        self.assertNotIn("①", cleaned_order)
        pos_b = cleaned_order.find("Instead of remembering")
        pos_a = cleaned_order.find("This externalization")
        pos_c = cleaned_order.find("Therefore, educators")
        self.assertTrue(0 < pos_b < pos_a < pos_c)

        raw_irrel_q = (
            "35. 다음 글에서 전체 흐름과 관계없는 문장은?\n"
            "Cognitive offloading refers to using physical action to reduce cognitive demand. "
            "① Instead of remembering a list of groceries, we write the items down on paper. "
            "② This externalization frees up our working memory capacity for complex reasoning. "
            "③ Grocery stores frequently discount seasonal fruits to attract weekend shoppers. "
            "④ However, excessive reliance on digital devices can weaken intrinsic memory pathways. "
            "⑤ Therefore, educators emphasize balancing convenience with deliberate mental practice."
        )
        cleaned_irrel = prepare_clean_english_source_passage(
            source_passage="",
            question_text=raw_irrel_q,
            q_num=35,
            orig_question_type="무관한 문장",
            orig_answer="3",
        )
        self.assertNotIn("Grocery stores frequently discount", cleaned_irrel)
        self.assertIn("Instead of remembering", cleaned_irrel)
        self.assertIn("Therefore, educators emphasize", cleaned_irrel)

    def test_compound_sentence_expansion_for_short_passage(self) -> None:
        short_3_sent = (
            "Scientific models simplify complex natural phenomena into manageable variables; however, they inevitably omit subtle local interactions that occur in real ecosystems. "
            "Researchers initially build idealized equations under strict laboratory assumptions, and they subsequently test these predictions against empirical field observations. "
            "When systematic discrepancies emerge between theoretical curves and measured data, scientists revise their foundational parameters, because accurate forecasting requires continuous empirical calibration."
        )
        norm_p, sents, spans = _prepare_indexed_sentences_for_transform(short_3_sent, "주어진 문장 삽입")
        self.assertGreaterEqual(len(sents), 6)
        self.assertEqual(len(spans), len(sents))
        for idx, (st, en) in enumerate(spans):
            self.assertEqual(norm_p[st:en], sents[idx])

    def test_assemble_gist_types(self) -> None:
        for gist_type in ("요지 파악", "주제 파악", "제목 추론"):
            norm_p, sents, spans = _prepare_indexed_sentences_for_transform(SAMPLE_6_SENT_PASSAGE, gist_type)
            spec = EnglishGistTransformSpec(
                choices=[
                    "Choice A for reading comprehension",
                    "Choice B for reading comprehension",
                    "Choice C for reading comprehension",
                    "Choice D for reading comprehension",
                    "Choice E for reading comprehension",
                ],
                correct_index=4,
                explanation="정답은 ①번입니다. 디지털 기기 의존과 기억력 균형이 핵심입니다.",
                key_terms=["offloading → 외재화", "resilience → 회복력"],
            )
            rw = _assemble_rewritten_question_from_spec(spec, gist_type, norm_p, sents, spans)
            issues = _validate_english_transformed_output(rw, gist_type, "밑줄형 (①~⑤)", source_passage=norm_p)
            self.assertEqual(issues, [])
            self.assertEqual(rw.preserved_answer, "4")
            self.assertIn("④번", rw.pdf_grounded_explanation)

    def test_assemble_blank_and_implied_meaning(self) -> None:
        norm_p, sents, spans = _prepare_indexed_sentences_for_transform(SAMPLE_6_SENT_PASSAGE, "빈칸 추론 (구)")
        spec_blank = EnglishSpanTargetTransformSpec(
            target_sentence_index=4,
            target_span="encoding the details deeply",
            correct_choice="in encoding the details deeply",
            distractor_choices=[
                "replacing all analog notebooks immediately",
                "memorizing unrelated historical dates",
                "avoiding collaborative group discussions",
                "upgrading wireless network hardware",
            ],
            explanation="정답은 ①번입니다. 온라인 접근성을 믿으면 깊은 부호화 노력을 덜 기울입니다.",
            key_terms=["encode → 부호화하다", "accessible → 접근 가능한"],
        )
        rw_blank = _assemble_rewritten_question_from_spec(
            spec_blank,
            "빈칸 추론 (구)",
            norm_p,
            sents,
            spans,
            blank_unit="구 (3~6단어)",
        )
        issues_blank = _validate_english_transformed_output(
            rw_blank,
            "빈칸 추론 (구)",
            "밑줄형 (①~⑤)",
            source_passage=norm_p,
        )
        self.assertEqual(issues_blank, [])
        self.assertIn("___________", rw_blank.rewritten_stem)
        ans_choice = rw_blank.rewritten_choices[int(rw_blank.preserved_answer) - 1]
        self.assertEqual(ans_choice, "encoding the details deeply")

        spec_imp = EnglishSpanTargetTransformSpec(
            target_sentence_index=0,
            target_span="Cognitive offloading",
            correct_choice="delegating mental storage tasks to external physical tools",
            distractor_choices=[
                "suppressing emotional reactions during academic exams",
                "eliminating all digital devices from modern classrooms",
                "competing with peers for faster calculation speed",
                "ignoring sensory feedback from the physical environment",
            ],
            explanation="외부 물리적 수단에 인지 부담을 위임하는 것을 뜻합니다.",
            key_terms=["offloading → 부담 경감"],
        )
        rw_imp = _assemble_rewritten_question_from_spec(
            spec_imp,
            "함축적 의미 추론",
            norm_p,
            sents,
            spans,
        )
        issues_imp = _validate_english_transformed_output(
            rw_imp,
            "함축적 의미 추론",
            "밑줄형 (①~⑤)",
            source_passage=norm_p,
        )
        self.assertEqual(issues_imp, [])
        self.assertIn("<u>Cognitive offloading</u>", rw_imp.rewritten_stem)

    def test_assemble_grammar_and_vocab_underline_five(self) -> None:
        norm_p, sents, spans = _prepare_indexed_sentences_for_transform(
            SAMPLE_6_SENT_PASSAGE,
            "어법상 틀린 것 (밑줄 ①~⑤)",
        )
        spec_gram = EnglishUnderlineFiveTransformSpec(
            items=[
                EnglishUnderlineFiveItem(
                    sentence_index=0,
                    original_span="using",
                    displayed_span="using",
                    is_error_target=False,
                ),
                EnglishUnderlineFiveItem(
                    sentence_index=3,
                    original_span="weaken",
                    displayed_span="weakening",
                    is_error_target=True,
                ),
                EnglishUnderlineFiveItem(
                    sentence_index=2,
                    original_span="frees",
                    displayed_span="frees",
                    is_error_target=False,
                ),
                EnglishUnderlineFiveItem(
                    sentence_index=4,
                    original_span="that",
                    displayed_span="that",
                    is_error_target=False,
                ),
                EnglishUnderlineFiveItem(
                    sentence_index=5,
                    original_span="balancing",
                    displayed_span="balancing",
                    is_error_target=False,
                ),
            ],
            explanation="조동사 can 뒤에는 동사원형 weaken이 와야 하므로 weakening은 어법상 틀립니다.",
            key_terms=["intrinsic → 내재적인"],
        )
        rw_gram = _assemble_rewritten_question_from_spec(
            spec_gram,
            "어법상 틀린 것 (밑줄 ①~⑤)",
            norm_p,
            sents,
            spans,
        )
        issues_gram = _validate_english_transformed_output(
            rw_gram,
            "어법상 틀린 것 (밑줄 ①~⑤)",
            "밑줄형 (①~⑤)",
            source_passage=norm_p,
        )
        self.assertEqual(issues_gram, [])
        self.assertEqual(rw_gram.preserved_answer, "3")
        self.assertIn("③ <u>weakening</u>", rw_gram.rewritten_stem)

        spec_vocab = EnglishUnderlineFiveTransformSpec(
            items=[
                EnglishUnderlineFiveItem(
                    sentence_index=0, original_span="reduce", displayed_span="reduce", is_error_target=False
                ),
                EnglishUnderlineFiveItem(
                    sentence_index=2, original_span="limited", displayed_span="limited", is_error_target=False
                ),
                EnglishUnderlineFiveItem(
                    sentence_index=3, original_span="weaken", displayed_span="strengthen", is_error_target=True
                ),
                EnglishUnderlineFiveItem(
                    sentence_index=4, original_span="accessible", displayed_span="accessible", is_error_target=False
                ),
                EnglishUnderlineFiveItem(
                    sentence_index=5, original_span="deliberate", displayed_span="deliberate", is_error_target=False
                ),
            ],
            explanation="외부 디지털 기기에 과도하게 의존하면 내재적 기억 인출 경로가 약화(weaken)되므로 strengthen은 부적절합니다.",
            key_terms=["reliance → 의존"],
        )
        rw_vocab = _assemble_rewritten_question_from_spec(
            spec_vocab,
            "문맥상 어휘 적절성 (밑줄 ①~⑤)",
            norm_p,
            sents,
            spans,
        )
        issues_vocab = _validate_english_transformed_output(
            rw_vocab,
            "문맥상 어휘 적절성 (밑줄 ①~⑤)",
            "밑줄형 (①~⑤)",
            source_passage=norm_p,
        )
        self.assertEqual(issues_vocab, [])
        self.assertEqual(rw_vocab.preserved_answer, "3")
        self.assertEqual(rw_vocab.rewritten_choices[2], "strengthen")

    def test_assemble_selective_vocab_abc(self) -> None:
        norm_p, sents, spans = _prepare_indexed_sentences_for_transform(
            SAMPLE_6_SENT_PASSAGE,
            "문맥상 어휘 적절성 (선택형 (A)(B)(C))",
        )
        spec_abc = EnglishVocabChoiceTransformSpec(
            items=[
                EnglishVocabChoiceTripleItem(sentence_index=0, original_word="reduce", antonym_word="increase"),
                EnglishVocabChoiceTripleItem(sentence_index=3, original_word="weaken", antonym_word="strengthen"),
                EnglishVocabChoiceTripleItem(sentence_index=4, original_word="less", antonym_word="more"),
            ],
            explanation="인지 부담을 줄이고(reduce), 과도한 의존은 기억 경로를 약화시키며(weaken), 부호화 노력을 덜(less) 기울입니다.",
            key_terms=["offloading → 외재화"],
        )
        rw_abc = _assemble_rewritten_question_from_spec(
            spec_abc,
            "문맥상 어휘 적절성 (선택형 (A)(B)(C))",
            norm_p,
            sents,
            spans,
            vocab_style="선택형 ((A)(B)(C) 네모)",
        )
        issues_abc = _validate_english_transformed_output(
            rw_abc,
            "문맥상 어휘 적절성 (선택형 (A)(B)(C))",
            "선택형 ((A)(B)(C) 네모)",
            source_passage=norm_p,
        )
        self.assertEqual(issues_abc, [])
        ans_choice = rw_abc.rewritten_choices[int(rw_abc.preserved_answer) - 1]
        self.assertEqual(ans_choice, "(A) reduce — (B) weaken — (C) less")

    def test_assemble_order_insertion_irrelevant_and_summary(self) -> None:
        norm_p, sents, spans = _prepare_indexed_sentences_for_transform(SAMPLE_6_SENT_PASSAGE, "글의 순서 ((A)-(B)-(C))")
        spec_ord = EnglishOrderTransformSpec(
            split_sentence_indices=[1, 3, 5],
            explanation="주어진 글 뒤에 구체적 예시와 역접, 결론이 이어집니다.",
            key_terms=["externalization → 외재화"],
        )
        rw_ord = _assemble_rewritten_question_from_spec(spec_ord, "글의 순서 ((A)-(B)-(C))", norm_p, sents, spans)
        self.assertEqual(
            _validate_english_transformed_output(rw_ord, "글의 순서 ((A)-(B)-(C))", "", source_passage=norm_p),
            [],
        )

        spec_ins = EnglishInsertionTransformSpec(
            extracted_sentence_index=3,
            explanation="However로 시작해 디지털 기기 과의존의 부작용으로 전환되는 자리입니다.",
            key_terms=["reliance → 의존"],
        )
        rw_ins = _assemble_rewritten_question_from_spec(spec_ins, "주어진 문장 삽입", norm_p, sents, spans)
        self.assertEqual(
            _validate_english_transformed_output(rw_ins, "주어진 문장 삽입", "", source_passage=norm_p),
            [],
        )

        spec_irr = EnglishIrrelevantTransformSpec(
            insert_position=3,
            unrelated_sentence="Modern smartphone manufacturers compete fiercely to increase camera resolution and battery longevity.",
            explanation="스마트폰 카메라 해상도와 배터리 경쟁은 인지 외재화 및 기억력과는 무관합니다.",
            key_terms=["longevity → 수명"],
        )
        rw_irr = _assemble_rewritten_question_from_spec(spec_irr, "전체 흐름과 무관한 문장", norm_p, sents, spans)
        self.assertEqual(
            _validate_english_transformed_output(rw_irr, "전체 흐름과 무관한 문장", "", source_passage=norm_p),
            [],
        )
        self.assertEqual(rw_irr.preserved_answer, "3")

        spec_sum = EnglishSummaryTransformSpec(
            summary_template="While externalizing information helps (A) ________ working memory load, overreliance on digital tools can (B) ________ internal memory retention.",
            correct_word_a="alleviate",
            correct_word_b="impair",
            distractor_pairs=[
                EnglishSummaryPairItem(word_a="alleviate", word_b="enhance"),
                EnglishSummaryPairItem(word_a="intensify", word_b="impair"),
                EnglishSummaryPairItem(word_a="intensify", word_b="stabilize"),
                EnglishSummaryPairItem(word_a="measure", word_b="promote"),
            ],
            explanation="정보를 외부화하면 작업 기억 부담을 완화(alleviate)하지만 과의존은 내재 기억을 손상(impair)시킬 수 있습니다.",
            key_terms=["alleviate → 완화하다", "impair → 손상시키다"],
        )
        rw_sum = _assemble_rewritten_question_from_spec(spec_sum, "요약문 완성 ((A), (B))", norm_p, sents, spans)
        self.assertEqual(
            _validate_english_transformed_output(rw_sum, "요약문 완성 ((A), (B))", "", source_passage=norm_p),
            [],
        )
        ans_sum_choice = rw_sum.rewritten_choices[int(rw_sum.preserved_answer) - 1]
        self.assertEqual(ans_sum_choice, "(A) alleviate — (B) impair")

    def test_end_to_end_generate_and_cache_v2_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = StorageManager(local_root=tmp_dir)
            mock_spec = EnglishVocabChoiceTransformSpec(
                items=[
                    EnglishVocabChoiceTripleItem(sentence_index=0, original_word="reduce", antonym_word="increase"),
                    EnglishVocabChoiceTripleItem(sentence_index=3, original_word="weaken", antonym_word="strengthen"),
                    EnglishVocabChoiceTripleItem(sentence_index=4, original_word="less", antonym_word="more"),
                ],
                explanation="문맥에 맞는 어휘 조합을 고릅니다.",
                key_terms=["reduce → 줄이다"],
            )
            with patch(
                "src.gemini.english_transform._generate_structured_with_fallback",
                return_value=mock_spec,
            ) as mock_gen:
                rw1 = generate_english_transformed_question(
                    api_key="test-key",
                    model_name="gemini-3.5-flash-lite",
                    source_passage=SAMPLE_6_SENT_PASSAGE,
                    target_type="문맥상 어휘 적절성",
                    vocab_style="선택형 ((A)(B)(C) 네모)",
                    storage=storage,
                    use_cache=True,
                    username="tester1",
                )
                self.assertEqual(mock_gen.call_count, 1)
                self.assertIn("(A) [", rw1.rewritten_stem)
                self.assertIn("(B) [", rw1.rewritten_stem)
                self.assertIn("(C) [", rw1.rewritten_stem)

                rw2 = generate_english_transformed_question(
                    api_key="test-key",
                    model_name="gemini-3.5-flash-lite",
                    source_passage=SAMPLE_6_SENT_PASSAGE,
                    target_type="문맥상 어휘 적절성",
                    vocab_style="선택형 ((A)(B)(C) 네모)",
                    storage=storage,
                    use_cache=True,
                    username="tester2",
                )
                self.assertEqual(mock_gen.call_count, 1)
                self.assertEqual(rw2.rewritten_stem, rw1.rewritten_stem)

    def test_duplicate_word_disambiguation_by_sentence_index(self) -> None:
        dup_passage = (
            "Researchers often model complex systems to predict future outcomes. "
            "However, a simple model cannot capture every variable in nature. "
            "When scientists refine their model with field data, accuracy improves significantly. "
            "Ultimately, each new model guides better policy decisions for environmental conservation. "
            "Therefore, continuous testing remains essential in modern science."
        )
        norm_p, sents, spans = _prepare_indexed_sentences_for_transform(dup_passage, "문맥상 어휘 적절성 (밑줄 ①~⑤)")
        # 'model'이 S0, S1, S2, S3에 모두 등장하지만 S2의 'refine'을 타겟으로, S3의 'model'을 4번 밑줄로 지정
        spec = EnglishUnderlineFiveTransformSpec(
            items=[
                EnglishUnderlineFiveItem(sentence_index=0, original_span="predict", displayed_span="predict", is_error_target=False),
                EnglishUnderlineFiveItem(sentence_index=1, original_span="capture", displayed_span="capture", is_error_target=False),
                EnglishUnderlineFiveItem(sentence_index=2, original_span="improves", displayed_span="deteriorates", is_error_target=True),
                EnglishUnderlineFiveItem(sentence_index=3, original_span="model", displayed_span="model", is_error_target=False),
                EnglishUnderlineFiveItem(sentence_index=4, original_span="essential", displayed_span="essential", is_error_target=False),
            ],
            explanation="현장 데이터로 모델을 정교화하면 정확도가 향상(improves)되므로 deteriorates는 부적절합니다.",
            key_terms=["refine → 정교화하다"],
        )
        rw = _assemble_rewritten_question_from_spec(spec, "문맥상 어휘 적절성 (밑줄 ①~⑤)", norm_p, sents, spans)
        self.assertIn("Ultimately, each new ④ <u>model</u> guides", rw.rewritten_stem)
        self.assertEqual(rw.preserved_answer, "3")

    def test_blank_verb_preservation_auto_trim(self) -> None:
        norm_p, sents, spans = _prepare_indexed_sentences_for_transform(SAMPLE_6_SENT_PASSAGE, "빈칸 추론 (구)")
        # S2("This externalization frees up our limited working memory capacity...")에서 본동사 'frees'까지 통째로 target_span에 넣은 경우
        spec = EnglishSpanTargetTransformSpec(
            target_sentence_index=2,
            target_span="frees up our limited working memory capacity",
            correct_choice="up our limited working memory capacity",
            distractor_choices=[
                "down our physical sensory perception",
                "out all long-term historical records",
                "away every social communication skill",
                "into strict mathematical calculations",
            ],
            explanation="외재화는 제한된 작업 기억 용량을 확보해 줍니다.",
            key_terms=["working memory → 작업 기억"],
        )
        rw = _assemble_rewritten_question_from_spec(spec, "빈칸 추론 (구)", norm_p, sents, spans)
        issues = _validate_english_transformed_output(rw, "빈칸 추론 (구)", "", source_passage=norm_p)
        self.assertEqual(issues, [])
        self.assertIn("This externalization frees ___________", rw.rewritten_stem)

    def test_invalid_spec_retries_and_blocks_polluted_return(self) -> None:
        # 5개 밑줄 중 아무것도 변형하지 않은 불량 스펙을 반환했을 때 3회 재시도 후 ValueError로 차단되고 캐시에 저장되지 않는지 검증
        bad_spec = EnglishUnderlineFiveTransformSpec(
            items=[
                EnglishUnderlineFiveItem(sentence_index=0, original_span="reduce", displayed_span="reduce", is_error_target=False),
                EnglishUnderlineFiveItem(sentence_index=1, original_span="write", displayed_span="write", is_error_target=False),
                EnglishUnderlineFiveItem(sentence_index=2, original_span="frees", displayed_span="frees", is_error_target=True),
                EnglishUnderlineFiveItem(sentence_index=3, original_span="weaken", displayed_span="weaken", is_error_target=False),
                EnglishUnderlineFiveItem(sentence_index=4, original_span="invest", displayed_span="invest", is_error_target=False),
            ],
            explanation="모두 원문 그대로인 잘못된 스펙",
            key_terms=[],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = StorageManager(local_root=tmp_dir)
            with patch(
                "src.gemini.english_transform._generate_structured_with_fallback",
                return_value=bad_spec,
            ) as mock_gen:
                with self.assertRaises(ValueError):
                    generate_english_transformed_question(
                        api_key="test-key",
                        model_name="gemini-3.5-flash-lite",
                        source_passage=SAMPLE_6_SENT_PASSAGE,
                        target_type="어법상 틀린 것 (밑줄 ①~⑤)",
                        storage=storage,
                    )
                self.assertEqual(mock_gen.call_count, 3)
                self.assertEqual(storage.load_english_pool(), {})


if __name__ == "__main__":
    unittest.main()
