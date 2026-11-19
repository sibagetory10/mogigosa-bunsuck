from __future__ import annotations

import base64
import html
import json
from datetime import datetime
from typing import Callable

from src.models import QuestionRecord


def build_exam_ai_jsonl(records: list[QuestionRecord]) -> str:
    """AI 서칭·문제 변형 시 토큰 소모를 최소화하도록 1문항 1줄 압축 JSONL 형식으로 직렬화한다."""
    ordered = sorted(records, key=lambda r: r.q_num)
    lines: list[str] = []
    for r in ordered:
        compact_obj: dict[str, object] = {
            "id": r.id,
            "q": r.q_num,
            "subj": r.curriculum_subject,
            "unit": r.unit_name,
            "gr": r.min_grade,
            "single": r.is_single_concept,
            "prin": r.core_principles,
            "terms": r.required_terms,
            "ans": r.answer,
            "exp": r.official_explanation,
        }
        q_txt = (getattr(r, "question_text", "") or "").strip()
        if q_txt:
            compact_obj["txt"] = q_txt
        std_code = (getattr(r, "standard_code", "") or "").strip()
        std_codes = list(getattr(r, "standard_codes", []) or [])
        if std_code and std_code not in std_codes:
            std_codes = [std_code, *std_codes]
        if std_code:
            compact_obj["std"] = std_code
        if len(std_codes) > 1:
            compact_obj["stds"] = std_codes
        adv_terms = list(getattr(r, "advanced_terms", []) or [])
        if adv_terms:
            compact_obj["adv_terms"] = adv_terms
        if r.subject_category == "영어":
            compact_obj["eng_sec"] = getattr(r, "english_section", "")
            compact_obj["eng_type"] = getattr(r, "english_question_type", "")
            compact_obj["topic"] = getattr(r, "passage_topic", "")
            compact_obj["vocab"] = list(getattr(r, "key_vocabulary", []) or [])
            if getattr(r, "passage_group_id", ""):
                compact_obj["group"] = r.passage_group_id
            if getattr(r, "passage_text", ""):
                compact_obj["passage"] = r.passage_text
        lines.append(json.dumps(compact_obj, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines)


def build_exam_markdown(
    exam_name: str,
    subject_category: str,
    records: list[QuestionRecord],
) -> str:
    """과목·모의고사 회차별 문항 DB를 읽기 쉬운 Markdown 문서로 직렬화한다."""
    ordered = sorted(records, key=lambda r: r.q_num)
    lines: list[str] = [
        f"# {exam_name}",
        "",
        f"- **대분류 과목:** {subject_category}",
        f"- **문항 수:** {len(ordered)}문항",
        f"- **생성 시각:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ]

    for rec in ordered:
        if rec.subject_category == "영어":
            sec = getattr(rec, "english_section", "") or ("듣기" if rec.q_num <= 17 else "독해")
            q_type = getattr(rec, "english_question_type", "") or rec.unit_name
            grp = getattr(rec, "passage_group_id", "")
            grp_suffix = f" · 세트 `[{grp}]`" if grp else ""
            topic = getattr(rec, "passage_topic", "") or "-"
            vocab = ", ".join(getattr(rec, "key_vocabulary", []) or []) or "-"
            passage = (getattr(rec, "passage_text", "") or "").strip()
            lines.extend(
                [
                    f"## {rec.q_num}번 ([{sec}] {q_type}{grp_suffix})",
                    "",
                    f"- **영역:** {sec} | **유형:** {q_type} | **정답:** `{rec.answer}`",
                    f"- **주제·소재:** {topic}",
                    f"- **핵심 어휘:** {vocab}",
                ]
            )
            if passage:
                lines.extend(["- **영어 지문 원문:**", "", f"> {passage.replace(chr(10), chr(10) + '> ')}", ""])
            lines.extend([f"- **공식 해설:** {rec.official_explanation or '-'}", ""])
            continue

        single_label = "단일 원리" if rec.is_single_concept else "복합 원리"
        principles = ", ".join(rec.core_principles) if rec.core_principles else "-"
        terms = ", ".join(rec.required_terms) if rec.required_terms else "-"
        adv_terms = list(getattr(rec, "advanced_terms", []) or [])
        basic_terms = list(getattr(rec, "basic_terms", []) or [])
        std_code = (getattr(rec, "standard_code", "") or "").strip()
        std_codes = [c for c in (getattr(rec, "standard_codes", []) or []) if c]
        if std_code and std_code not in std_codes:
            std_codes = [std_code, *std_codes]
        grade_ev = (getattr(rec, "grade_evidence", "") or "").strip()
        std_suffix = (
            " · 성취기준 " + ", ".join(f"`[{c}]`" for c in std_codes)
            if std_codes
            else ""
        )
        lines.extend(
            [
                f"## {rec.q_num}번 ({rec.curriculum_subject} · {rec.unit_name}{std_suffix})",
                "",
                f"- **최저 학년:** 고{rec.min_grade} | **개념 유형:** {single_label} | **정답:** `{rec.answer}`",
                f"- **본질 원리:** {principles}",
                f"- **표면 용어(전체):** {terms}",
            ]
        )
        if adv_terms or basic_terms:
            lines.append(
                f"- **용어 구분:** 상위 전문 용어({', '.join(adv_terms) or '-'}) / 해당 학년 기본 용어({', '.join(basic_terms) or '-'})"
            )
        if grade_ev:
            lines.append(f"- **학년 판정 근거:** {grade_ev}")
        lines.extend(
            [
                f"- **공식 해설:** {rec.official_explanation or '-'}",
                "",
            ]
        )
    return "\n".join(lines)


def build_exam_html(
    exam_name: str,
    subject_category: str,
    records: list[QuestionRecord],
    crop_loader: Callable[[str], bytes | None] | None = None,
) -> str:
    """과목·모의고사 회차별 문항 DB를 독립 실행형 HTML 웹페이지로 생성한다."""
    ordered = sorted(records, key=lambda r: r.q_num)
    cards_html: list[str] = []

    for rec in ordered:
        img_tag = '<div class="no-img">크롭 이미지 없음</div>'
        if crop_loader and rec.image_filename:
            png_bytes = crop_loader(rec.image_filename)
            if png_bytes:
                b64 = base64.b64encode(png_bytes).decode("ascii")
                img_tag = f'<img src="data:image/png;base64,{b64}" alt="Q{rec.q_num}" loading="lazy" />'

        principles_html = "".join(
            f'<span class="chip chip-blue">{html.escape(p)}</span>'
            for p in rec.core_principles
        ) or '<span class="muted">-</span>'
        terms_html = "".join(
            f'<span class="chip chip-amber">{html.escape(t)}</span>'
            for t in rec.required_terms
        ) or '<span class="muted">-</span>'
        adv_terms = list(getattr(rec, "advanced_terms", []) or [])
        basic_terms = list(getattr(rec, "basic_terms", []) or [])
        std_code = (getattr(rec, "standard_code", "") or "").strip()
        std_codes = [c for c in (getattr(rec, "standard_codes", []) or []) if c]
        if std_code and std_code not in std_codes:
            std_codes = [std_code, *std_codes]
        grade_ev = (getattr(rec, "grade_evidence", "") or "").strip()

        std_badge = "".join(f'<span class="pill">[{html.escape(c)}]</span>' for c in std_codes)
        ev_html = (
            f'<div class="field-label">학년 판정 근거</div><p class="expl">{html.escape(grade_ev)}</p>'
            if grade_ev
            else ""
        )
        split_terms_html = ""
        if adv_terms or basic_terms:
            adv_str = ", ".join(html.escape(t) for t in adv_terms) or "-"
            bas_str = ", ".join(html.escape(t) for t in basic_terms) or "-"
            split_terms_html = (
                f'<div class="muted" style="margin-bottom:6px;">'
                f'상위 전문: {adv_str} · 기본 용어: {bas_str}</div>'
            )

        if rec.subject_category == "영어":
            sec = getattr(rec, "english_section", "") or ("듣기" if rec.q_num <= 17 else "독해")
            q_type = getattr(rec, "english_question_type", "") or rec.unit_name
            grp = getattr(rec, "passage_group_id", "")
            grp_badge = f'<span class="pill">세트 [{html.escape(grp)}]</span>' if grp else ""
            topic = getattr(rec, "passage_topic", "") or "-"
            vocab_html = "".join(
                f'<span class="chip chip-amber">{html.escape(v)}</span>'
                for v in (getattr(rec, "key_vocabulary", []) or [])
            ) or '<span class="muted">-</span>'
            passage_txt = (getattr(rec, "passage_text", "") or "").strip()
            passage_html = (
                f'<div class="field-label">영어 지문 원문 (passage_text)</div>'
                f'<p class="expl" style="white-space:pre-wrap; font-family:Georgia,serif;">{html.escape(passage_txt)}</p>'
                if passage_txt
                else ""
            )
            cards_html.append(
                f"""
                <article class="card" id="q-{rec.q_num}">
                  <header class="card-head">
                    <div class="q-title">
                      <span class="q-num">{rec.q_num}번</span>
                      <span class="badge">{html.escape(sec)}</span>
                      <span class="unit">{html.escape(q_type)}</span>
                    </div>
                    <div class="meta-pills">
                      {grp_badge}
                      <span class="pill pill-ans">정답 {html.escape(rec.answer)}</span>
                    </div>
                  </header>
                  <div class="card-body">
                    <div class="img-col">{img_tag}</div>
                    <div class="info-col">
                      <div class="field-label">주제 · 소재 (passage_topic)</div>
                      <p class="expl">{html.escape(topic)}</p>
                      <div class="field-label">핵심 어휘 (key_vocabulary)</div>
                      <div class="chips">{vocab_html}</div>
                      {passage_html}
                      <div class="field-label">공식 해설</div>
                      <p class="expl">{html.escape(rec.official_explanation or "-")}</p>
                    </div>
                  </div>
                </article>
                """
            )
            continue

        single_txt = "단일 원리" if rec.is_single_concept else "복합 원리"
        cards_html.append(
            f"""
            <article class="card" id="q-{rec.q_num}">
              <header class="card-head">
                <div class="q-title">
                  <span class="q-num">{rec.q_num}번</span>
                  <span class="badge">{html.escape(rec.curriculum_subject)}</span>
                  <span class="unit">{html.escape(rec.unit_name)}</span>
                </div>
                <div class="meta-pills">
                  {std_badge}
                  <span class="pill">고{rec.min_grade}</span>
                  <span class="pill">{single_txt}</span>
                  <span class="pill pill-ans">정답 {html.escape(rec.answer)}</span>
                </div>
              </header>
              <div class="card-body">
                <div class="img-col">{img_tag}</div>
                <div class="info-col">
                  <div class="field-label">본질 원리 (core_principles)</div>
                  <div class="chips">{principles_html}</div>
                  <div class="field-label">표면 용어 (required_terms)</div>
                  {split_terms_html}
                  <div class="chips">{terms_html}</div>
                  {ev_html}
                  <div class="field-label">공식 해설</div>
                  <p class="expl">{html.escape(rec.official_explanation or "-")}</p>
                </div>
              </div>
            </article>
            """
        )

    nav_links = "".join(
        f'<a class="q-link" href="#q-{r.q_num}">{r.q_num}번</a>'
        for r in ordered
    )

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(exam_name)} · 기출 DB</title>
  <style>
    :root {{
      --bg: #f6f8fc; --card: #ffffff; --border: #e2e8f0;
      --text: #0f172a; --muted: #64748b;
      --blue-bg: #dbeafe; --blue-fg: #1d4ed8;
      --amber-bg: #fef3c7; --amber-fg: #92400e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 28px 18px 60px; background: var(--bg);
      color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Malgun Gothic", sans-serif;
    }}
    .wrap {{ max-width: 1080px; margin: 0 auto; }}
    .hero {{
      background: linear-gradient(135deg, #0f172a, #1e3a8a); color: #fff;
      border-radius: 16px; padding: 24px 28px; margin-bottom: 18px;
    }}
    .hero h1 {{ margin: 0 0 8px; font-size: 1.55rem; }}
    .hero p {{ margin: 0; opacity: .9; font-size: .95rem; }}
    .nav {{
      display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 20px;
      background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 12px;
    }}
    .q-link {{
      text-decoration: none; font-size: .82rem; font-weight: 600;
      padding: 4px 10px; border-radius: 999px; background: #eff6ff; color: #1d4ed8;
    }}
    .q-link:hover {{ background: #dbeafe; }}
    .card {{
      background: var(--card); border: 1px solid var(--border); border-radius: 14px;
      margin-bottom: 16px; overflow: hidden; box-shadow: 0 2px 8px rgba(15,23,42,.04);
    }}
    .card-head {{
      display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
      gap: 10px; padding: 14px 18px; background: #f8fafc; border-bottom: 1px solid var(--border);
    }}
    .q-title {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }}
    .q-num {{ font-size: 1.05rem; font-weight: 800; color: #1e3a8a; }}
    .badge {{
      font-size: .78rem; font-weight: 700; padding: 3px 9px; border-radius: 999px;
      background: #e0e7ff; color: #3730a3;
    }}
    .unit {{ font-size: .92rem; font-weight: 600; color: #334155; }}
    .meta-pills {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .pill {{
      font-size: .78rem; font-weight: 600; padding: 3px 9px; border-radius: 999px;
      background: #f1f5f9; color: #334155;
    }}
    .pill-ans {{ background: #dcfce7; color: #166534; }}
    .card-body {{
      display: grid; grid-template-columns: minmax(260px, 1fr) 1.25fr; gap: 18px; padding: 18px;
    }}
    @media (max-width: 780px) {{ .card-body {{ grid-template-columns: 1fr; }} }}
    .img-col img {{ width: 100%; height: auto; border: 1px solid var(--border); border-radius: 10px; }}
    .no-img {{
      padding: 28px; text-align: center; color: var(--muted);
      border: 1px dashed var(--border); border-radius: 10px;
    }}
    .field-label {{ font-size: .82rem; font-weight: 700; color: #475569; margin: 10px 0 6px; }}
    .field-label:first-child {{ margin-top: 0; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }}
    .chip {{
      font-size: .8rem; font-weight: 600; padding: 3px 10px; border-radius: 999px;
    }}
    .chip-blue {{ background: var(--blue-bg); color: var(--blue-fg); }}
    .chip-amber {{ background: var(--amber-bg); color: var(--amber-fg); }}
    .expl {{ margin: 4px 0 0; line-height: 1.55; font-size: .92rem; color: #1e293b; white-space: pre-wrap; }}
    .muted {{ color: var(--muted); font-size: .84rem; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>📘 {html.escape(exam_name)}</h1>
      <p>과목: {html.escape(subject_category)} · 총 {len(ordered)}문항 · 생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </section>
    <nav class="nav">{nav_links}</nav>
    {''.join(cards_html)}
  </div>
</body>
</html>
"""
