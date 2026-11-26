from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pymupdf

from src.gemini.korean import (
    _extract_korean_passage_markers,
    _validate_and_sanitize_korean_question,
    balance_korean_answer_distribution,
    generate_korean_set_reconstruction,
)
from src.models import KoreanSetQuestionOutput, QuestionRecord, RewrittenQuestion
from src.pdf.exam_pdf import generate_custom_exam_pdf
from src.storage import StorageManager
from src.text_formatter import (
    clean_korean_exam_passage,
    render_korean_inline_html,
    split_stem_blocks,
)


def _make_korean_record(q_num: int, group_id: str, subgenre: str, passage_text: str) -> QuestionRecord:
    return QuestionRecord(
        id=f"2026_10_고1_국어_{q_num}",
        exam_name="2026년 10월 고1 국어",
        q_num=q_num,
        subject_category="국어",
        curriculum_subject=f"독서({subgenre})",
        unit_name=subgenre,
        min_grade=1,
        is_single_concept=False,
        core_principles=["지문 핵심 원리"],
        required_terms=[],
        answer="1",
        official_explanation="해설",
        passage_group_id=group_id,
        passage_text=passage_text,
    )


class KoreanTransformAndPdfTests(unittest.TestCase):
    def test_clean_korean_exam_passage_strips_noise_and_preserves_markers(self) -> None:
        raw = (
            "[11 ~ 14] 다음 글을 읽고 물음에 답하시오.\n"
            "4\n"
            "(가)\n"
            "자기계약과 쌍방\n"
            "국어 영역\n"
            "대리는 원칙적으로 금지된다. 아렌트는 제2차 세계 대전 전범의 재판을 분석\n"
            "5\n"
            "하며, ㉠전동기의 회전 속도를 제한하는 역기전력이 ⓐ따를 수 있다.\n"
            "보상청구권이\n"
            "7\n"
            "인정된다. 학생은 이를 이해할 수\n"
            "있다.\n"
            "(나)\n"
            "[A] 상인은 ⓑ보조상을 둔다."
        )
        cleaned = clean_korean_exam_passage(raw, is_poetry_or_dialogue=False)

        self.assertNotIn("[11 ~ 14]", cleaned)
        self.assertNotIn("다음 글을 읽고 물음에 답하시오", cleaned)
        self.assertNotIn("국어 영역", cleaned)
        self.assertNotIn("\n4\n", cleaned)
        self.assertNotIn("\n5\n", cleaned)
        self.assertNotIn("\n7\n", cleaned)
        self.assertIn("쌍방대리는 원칙적으로 금지된다.", cleaned)
        self.assertIn("재판을 분석하며,", cleaned)
        self.assertIn("보상청구권이 인정된다.", cleaned)
        self.assertIn("이해할 수 있다.", cleaned)
        for mk in ("(가)", "(나)", "[A]", "㉠", "ⓐ", "ⓑ"):
            self.assertIn(mk, cleaned)

    def test_bogi_prose_box_and_inline_markup(self) -> None:
        stem_with_prose_bogi = (
            "윗글의 내용을 바탕으로 <보기>의 상황을 분석한 것으로 가장 적절한 것은?\n\n"
            "<보기>\n"
            "갑은 주식 시장에 대해 높은 전문성을 지닌 을에게 자신의 주식 매매를 위탁하였다. "
            "을은 갑이 지정한 가격보다 불리한 가격으로 거래하게 되었다."
        )
        blocks = split_stem_blocks(stem_with_prose_bogi)
        kinds = [k for k, _ in blocks]
        self.assertIn("text", kinds)
        self.assertIn("bogi_prose", kinds)

        rendered = render_korean_inline_html("예상치 못한 손실이 **따를** 수 있다.", bold_to_underline=True)
        self.assertNotIn("**", rendered)
        self.assertIn("<u class='kor-u'>따를</u>", rendered)

    def test_korean_gate1_validation_catches_defects_and_sanitizes(self) -> None:
        passage = "(가) 상인은 보조상을 둔다. 대리상은 독립된 상인이다. ⓐ따를 수 있다."
        allowed = _extract_korean_passage_markers(passage)

        bad_q = KoreanSetQuestionOutput(
            slot_index=1,
            question_type_label="문맥 어휘",
            stem="<보기>\n갑이 을에게 매매를 위탁하였다.\n윗글의 ⓐ('따를 수 있다')와 <보기>를 바탕으로 보상청구권을 청구할 때 ㉮에 대한 설명으로 적절한 것은?",
            bogi_text="",
            choices=[
                "민법상 대리인은 독립된 상인이 아니지만, 상법상 체약대리상은 독립된 상인이 아니라는 점에서 차이가 난다.",
                "손실이 **따를** 수 있다.",
                "정상 선지 3이다.",
                "정상 선지 4이다.",
                "정상 선지 5이다.",
            ],
            answer="1",
            explanation="정답 해설이다.",
            choice_explanations=["1", "2", "3", "4", "5"],
        )
        sanitized, errs = _validate_and_sanitize_korean_question(bad_q, passage, allowed)
        self.assertEqual(sanitized.bogi_text, "갑이 을에게 매매를 위탁하였다.")
        self.assertTrue(sanitized.stem.startswith("윗글의 ⓐ와 <보기>를 바탕으로 보상을 청구할 때"))
        self.assertNotIn("ⓐ('따를 수 있다')", sanitized.stem)
        self.assertNotIn("보상청구권을 청구", sanitized.stem)
        self.assertIn("<u>따를</u>", sanitized.choices[1])
        self.assertNotIn("**", sanitized.choices[1])
        self.assertTrue(any("지문에 없는 표지 참조" in e and "㉮" in e for e in errs))
        self.assertTrue(any("자체모순" in e for e in errs))

        trap_q = KoreanSetQuestionOutput(
            slot_index=2,
            question_type_label="<보기> 적용",
            stem="윗글을 바탕으로 <보기>에서 아렌트가 강조한 ‘확장된 사유’ 및 ‘무사유’의 관점과 가장 잘 부합하는 것은?",
            bogi_text="갑은 몰두하고 있었다(ⓐ). 판단하지 않았고(ⓑ), 은폐하고 있었다(ⓒ). 을은 추론하며(ⓓ), 상상하고 있다고 평가했다(ⓔ).",
            choices=["ⓐ", "ⓑ", "ⓒ", "ⓓ", "ⓔ"],
            answer="4",
            explanation="해설",
            choice_explanations=["1", "2", "3", "4", "5"],
        )
        _, trap_errs = _validate_and_sanitize_korean_question(trap_q, passage, allowed)
        self.assertTrue(any("단일 기호 선지" in e for e in trap_errs))

    def test_balance_korean_answer_distribution_balances_choices_and_explanations(self) -> None:
        pairs: list[tuple[QuestionRecord, RewrittenQuestion]] = []
        for i in range(1, 11):
            rec = _make_korean_record(i, "1~10", "과학·기술", "지문 본문")
            exp = (
                "[정답 해설]\n4번 선지가 옳다.\n\n"
                "[선지별 분석]\n"
                "① 오답 A다.\n"
                "② 오답 B다.\n"
                "③ 오답 C다.\n"
                "④ 정답 D다.\n"
                "⑤ 오답 E다."
            )
            rw = RewrittenQuestion(
                original_question_id=rec.id,
                rewritten_stem=f"문항 {i} 발문이다.",
                rewritten_choices=["선지A", "선지B", "선지C", "정답선지D", "선지E"],
                pdf_grounded_explanation=exp,
                preserved_answer="4",
                self_verification_note="ok",
                target_question_type="내용 일치",
                subject_domain="국어",
                shared_passage="지문 본문",
                passage_group_key="exam::1~10",
            )
            pairs.append((rec, rw))

        balanced = balance_korean_answer_distribution(pairs)
        answers = [rw.preserved_answer for _, rw in balanced]
        counts = Counter(answers)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        for k in range(2, len(answers)):
            self.assertFalse(answers[k] == answers[k - 1] == answers[k - 2])

        circled = {"1": "①", "2": "②", "3": "③", "4": "④", "5": "⑤"}
        for _, rw in balanced:
            ans_idx = int(rw.preserved_answer) - 1
            self.assertEqual(rw.rewritten_choices[ans_idx], "정답선지D")
            sym = circled[rw.preserved_answer]
            self.assertIn(f"{sym} 정답 D다.", rw.pdf_grounded_explanation)

    def test_korean_3sets_13questions_pdf_end_to_end_no_clipping_and_clean_layout(self) -> None:
        p1 = (
            "자기장은 자기력이 작용하고 자기장 에너지가 존재하는 공간이다. 자기력선은 자석 외부에서 N극에서 S극으로 향하며 서로 겹치거나 끊어지지 않는다.\n\n"
            "영구자석 동기전동기는 영구자석과 전자석 사이의 인력과 척력을 이용해 전기 에너지를 회전력으로 바꾸는 장치이다. "
            "고정자의 안쪽에는 수십 개의 슬롯이 파여 있고 세 쌍의 구역으로 나뉜다.\n\n"
            "자동차가 가속 신호를 받으면 오른나사 법칙에 따라 전선 다발 주변에 자기장이 형성된다. "
            "고정자의 세 쌍의 슬롯 구역에 시간차를 두고 교류 전류를 공급하면 마치 하나의 전자석이 회전하는 것과 같은 효과가 나타난다. "
            "회전력을 가장 효율적으로 얻기 위해 영구자석 N극이 회전하는 방향의 90도 앞선 위치에 전자석 S극을 형성한다.\n\n"
            "자동차가 특정 속도 이상으로 주행하면 회전자가 빠르게 회전하여 ㉠역기전력이 커지게 된다. "
            "이를 극복하기 위해 영구자석의 N극과 전자석의 S극이 같은 방향에 놓이도록 전류를 제어하여 자기장 에너지를 상쇄한다."
        )
        p2 = (
            "칸트는 인간이 감성적 존재이면서 이성적 존재라고 보았다. 인간은 도덕법칙보다 욕망을 우선시하는 근본악의 성향을 지닌다.\n\n"
            "그러나 한나 아렌트는 현대 사회의 대량 학살 같은 악을 설명하는 데 칸트 윤리관이 한계가 있다고 지적했다. "
            "전범은 특별한 악의가 아니라 사유를 중단한 무사유 상태에서 범죄를 저질렀으며, 아렌트는 이를 악의 평범성이라 불렀다.\n\n"
            "아렌트에 따르면 인식은 지식을 습득하는 활동이고, 사유는 도덕적 의미를 끊임없이 되묻는 활동이다. "
            "이러한 무사유는 분업화된 업무 구조, 위계적 조직, 관행적 상투어에 의해 심화된다. "
            "이를 극복하기 위해 타자의 관점에서 세계를 이성적으로 추론하는 확장된 사유가 필요하다."
        )
        p3 = (
            "(가)\n"
            "민법은 사적 자치의 원칙을 기초로 대리 제도를 둔다. 대리에서는 법률효과가 귀속되는 본인, 의사표시를 대신하는 대리인, 거래 당사자인 상대방이 있다. "
            "임의대리는 수권행위에 의해 대리권이 발생하며 현명주의를 원칙으로 한다. "
            "자기계약과 쌍방대리는 원칙적으로 금지되지만, 기존의 의무를 실현하거나 본인이 허락한 경우에는 예외적으로 허용된다.\n\n"
            "(나)\n"
            "상법에서는 상인의 영업활동을 보조하는 보조상 제도를 규정하여 비용과 위험 부담이 ⓐ따를 수 있는 거래의 안전성을 도모한다. "
            "대리상은 특정 상인을 위해 지속적으로 대리 또는 중개를 수행하며 체약대리상과 중개대리상으로 나뉜다. "
            "위탁매매인은 자기 명의로써 타인의 계산으로 매매를 하는 독립된 상인이다. "
            "위탁매매인이 지정 가격보다 불리한 가격으로 거래하면 그 차액을 부담해야 거래 결과가 위탁자에게 귀속된다. "
            "대리상에게는 계약 종료 후에도 본인에게 이익이 남을 때 일정 기간 내에 보상을 청구할 수 있는 보상청구권이 인정된다."
        )

        items: list[dict] = []
        specs = [
            (11, 4, "11~14", "과학·기술", p1),
            (15, 3, "15~17", "인문·예술", p2),
            (18, 6, "18~23", "사회·경제", p3),
        ]
        global_idx = 0
        for start_q, count, gid, subg, p_txt in specs:
            gkey = f"2026년 10월 고1 국어::{gid}"
            for offset in range(count):
                global_idx += 1
                qn = start_q + offset
                rec = _make_korean_record(qn, gid, subg, p_txt)
                ans = str(((global_idx - 1) % 5) + 1)
                stem = f"제{global_idx}문항 고유발문확인용({subg}) 가장 적절한 것은?"
                if global_idx in (7, 12):
                    stem += (
                        "\n\n<보기>\n"
                        f"갑은 제{global_idx}문항 사례에서 자신의 거래를 을에게 위탁하였고, 을은 불리한 가격으로 거래하였다."
                    )
                choices = [
                    f"제{global_idx}문항 선지 1번 진술이다.",
                    f"제{global_idx}문항 선지 2번 진술이다.",
                    f"제{global_idx}문항 선지 3번 진술이다.",
                    f"제{global_idx}문항 선지 4번 진술이다.",
                    f"제{global_idx}문항 선지 5번 **따를** 진술이다.",
                ]
                rw = RewrittenQuestion(
                    original_question_id=rec.id,
                    rewritten_stem=stem,
                    rewritten_choices=choices,
                    pdf_grounded_explanation=(
                        f"[정답 해설]\n제{global_idx}문항 정답 근거다.\n\n"
                        "[선지별 분석]\n① 1번이다.\n② 2번이다.\n③ 3번이다.\n④ 4번이다.\n⑤ 5번이다."
                    ),
                    preserved_answer=ans,
                    self_verification_note="ok",
                    target_question_type="내용 일치" if global_idx not in (7, 12) else "<보기> 적용",
                    subject_domain="국어",
                    shared_passage=p_txt,
                    passage_group_key=gkey,
                    set_instruction="다음 글을 읽고 물음에 답하시오.",
                )
                items.append(
                    {
                        "question": rec,
                        "judgment": None,
                        "crop_png_bytes": None,
                        "rewritten": rw,
                        "concept_reference": f"출처: 2026년 10월 고1 국어 [{gid}번]",
                        "korean_meta": {
                            "section": "독서",
                            "subgenre": subg,
                            "group_id": gid,
                            "passage_text": p_txt,
                        },
                    }
                )

        pdf_bytes = generate_custom_exam_pdf(
            items=items,
            exam_title="수능·모의고사 국어(독서·문학) 지문 세트 변형 모의고사",
            source_label="",
            two_column=True,
            include_original_for_rewritten=False,
        )
        self.assertGreater(len(pdf_bytes), 5000)

        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            full_text = "\n".join(doc[i].get_text("text") for i in range(doc.page_count))

        self.assertIn("[1~4] 다음 글을 읽고 물음에 답하시오.", full_text)
        self.assertIn("[5~7] 다음 글을 읽고 물음에 답하시오.", full_text)
        self.assertIn("[8~13] 다음 글을 읽고 물음에 답하시오.", full_text)

        self.assertNotIn("[11 ~ 14]", full_text)
        self.assertNotIn("[15 ~ 17]", full_text)
        self.assertNotIn("[18 ~ 23]", full_text)
        self.assertNotIn("PDF 맞춤 개조", full_text)
        self.assertNotIn("기준 개념서:", full_text)
        self.assertNotIn("**", full_text)

        self.assertIn("〈 보 기 〉", full_text)

        for q_idx in range(1, 14):
            self.assertIn(f"제{q_idx}문항 고유발문확인용", full_text)
            for c_idx in range(1, 6):
                self.assertIn(f"제{q_idx}문항 선지 {c_idx}번", full_text)

    def test_korean_pool_store_cache_history_and_admin_ops(self) -> None:
        raw_noisy = (
            "[11 ~ 14] 다음 글을 읽고 물음에 답하시오.\n"
            "4\n"
            "자기장은 자기력이 작용하고 자기장 에너지가 존재하는 공간이다. 전동기는 인력과 척력을 이용한다.\n"
            "국어 영역\n"
            "고정자 표면에 전자석이 형성된다."
        )
        cleaned = clean_korean_exam_passage(raw_noisy, is_poetry_or_dialogue=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            storage = StorageManager(local_root=root / "exam_db", concept_root=root / "concept_db")

            key_raw = storage.build_korean_pool_key(raw_noisy, group_id="[11~14]", korean_section="독서", question_count=2)
            key_clean = storage.build_korean_pool_key(cleaned, group_id="11~14", korean_section="독서", question_count=2)
            self.assertEqual(key_raw, key_clean)

            def _make_set(tag: str) -> list[RewrittenQuestion]:
                return [
                    RewrittenQuestion(
                        original_question_id=f"2026_10_고1_국어_{11 + i}",
                        rewritten_stem=f"[{tag}] 문항 {i + 1} 발문이다.",
                        rewritten_choices=["선지1", "선지2", "선지3", "선지4", "선지5"],
                        pdf_grounded_explanation="정답 해설이다.",
                        preserved_answer=str((i % 5) + 1),
                        self_verification_note="ok",
                        target_question_type="내용 일치",
                        subject_domain="국어",
                        shared_passage=cleaned,
                        passage_group_key="2026년 10월 고1 국어::11~14",
                        set_instruction="다음 글을 읽고 물음에 답하시오.",
                    )
                    for i in range(2)
                ]

            # 최대 max_per_key=5 상한 및 중복 variant_id 방지 검증
            vids: list[str] = []
            for idx in range(6):
                v = storage.save_cached_korean_set(
                    key_clean,
                    _make_set(f"V{idx}"),
                    exam_name="2026년 10월 고1 국어",
                    group_id="11~14",
                    korean_section="독서",
                    korean_subgenre="과학·기술",
                    passage_topic="동기전동기 원리",
                    max_per_key=5,
                )
                vids.append(v)
            # 동일 세트 재저장 시 중복 추가되지 않아야 함
            dup_v = storage.save_cached_korean_set(
                key_clean,
                _make_set("V5"),
                exam_name="2026년 10월 고1 국어",
                group_id="11~14",
                korean_section="독서",
                max_per_key=5,
            )
            self.assertEqual(dup_v, vids[5])
            self.assertEqual(storage.get_korean_pool_set_count(key_clean), 5)

            # 신규 사용자는 즉시 캐시 히트
            hit1 = storage.get_cached_korean_set(key_clean, expected_count=2, username="alice")
            self.assertIsNotNone(hit1)
            _, hit_vid1 = hit1
            storage.mark_user_seen_korean_sets("alice", [hit_vid1])

            # 이미 본 세트를 모두 마킹하면 미열람 세트가 남아있는 동안 다른 세트를 반환하고, 모두 소진 시 None 반환
            remaining_vids = [v for v in vids[1:] if v != hit_vid1]
            for rv in remaining_vids:
                hit_next = storage.get_cached_korean_set(key_clean, expected_count=2, username="alice")
                self.assertIsNotNone(hit_next)
                self.assertNotEqual(hit_next[1], hit_vid1)
                storage.mark_user_seen_korean_sets("alice", [hit_next[1]])
            self.assertIsNone(storage.get_cached_korean_set(key_clean, expected_count=2, username="alice"))

            # 다른 사용자(bob)는 풀 캐시를 즉시 히트하고, generate_korean_set_reconstruction도 API 호출 없이 반환
            recs = [
                _make_korean_record(11, "11~14", "과학·기술", raw_noisy),
                _make_korean_record(12, "11~14", "과학·기술", raw_noisy),
            ]
            with patch("src.gemini.korean._build_client", side_effect=AssertionError("API 호출되면 안 됨")):
                out_pairs = generate_korean_set_reconstruction(
                    api_key="dummy",
                    model_name="gemini-3.5-flash-lite",
                    source_records=recs,
                    passage_text=raw_noisy,
                    korean_section="독서",
                    korean_subgenre="과학·기술",
                    storage=storage,
                    use_cache=True,
                    username="bob",
                )
            self.assertEqual(len(out_pairs), 2)
            self.assertEqual(out_pairs[0][0].id, recs[0].id)

            # 관리자 개별 삭제 / 회차명 변경 / 회차 삭제 / 전체 비우기 검증
            self.assertEqual(storage.delete_korean_pool_entry(key_clean, 0), 1)
            self.assertEqual(storage.get_korean_pool_set_count(key_clean), 4)
            self.assertEqual(storage.rename_korean_pool_exam("2026년 10월 고1 국어", "2026년 10월 학평 국어"), 4)
            self.assertEqual(storage.delete_korean_pool_by_exam("2026년 10월 학평 국어"), 4)
            self.assertEqual(storage.get_korean_pool_set_count(key_clean), 0)

            storage.save_cached_korean_set(key_clean, _make_set("FINAL"), exam_name="테스트", group_id="1~2")
            self.assertEqual(storage.clear_korean_pool(), 1)
            self.assertEqual(storage.load_korean_pool(), {})


if __name__ == "__main__":
    unittest.main()
