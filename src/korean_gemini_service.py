"""Backwards-compatible facade for Korean Gemini services.

Implementation now lives in ``src.gemini.korean``.
"""

from __future__ import annotations

from src.gemini.core import (  # noqa: F401
    _build_client,
    _generate_content_with_fallback,
    _parse_structured,
)
from src.gemini.korean import (  # noqa: F401
    _KOREAN_DIAGRAM_GUIDE,
    _KOREAN_LIT_DIAGRAM_GUIDE,
    _is_korean_poetry_genre,
    _sanitize_korean_analysis_output,
    _strip_korean_exam_noise_lines,
    _sync_korean_group_passages,
    analyze_korean_exam_pdf_batch_with_gemini,
    analyze_korean_question_with_gemini,
    fetch_korean_literature_web_brief,
    generate_korean_passage_analysis,
    generate_korean_set_reconstruction,
    generate_korean_visual_diagram,
    proofread_korean_passage_analysis,
)
