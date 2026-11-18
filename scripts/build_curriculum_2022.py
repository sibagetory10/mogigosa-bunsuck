from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.curriculum_2022 import (  # noqa: E402
    get_catalog_summary_stats,
    merge_ncic_standards_to_catalog,
    parse_ncic_curriculum_pdf,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NCIC(국가교육과정정보센터) 2022 개정 교육과정 별책 PDF를 파싱해 data/curriculum_2022.json을 갱신합니다."
    )
    parser.add_argument("--math", type=Path, help="별책 8 수학과 교육과정 PDF 경로")
    parser.add_argument("--science", type=Path, help="별책 9 과학과 교육과정 PDF 경로")
    parser.add_argument("--social", type=Path, help="별책 7 사회과 교육과정 PDF 경로")
    args = parser.parse_args()

    targets: list[tuple[str, Path | None]] = [
        ("수학", args.math),
        ("과학", args.science),
        ("사회", args.social),
    ]
    all_extracted = []
    for cat, pdf_path in targets:
        if pdf_path is None:
            continue
        if not pdf_path.exists():
            print(f"[ERROR] 파일을 찾을 수 없습니다: {pdf_path}", file=sys.stderr)
            return 1
        items = parse_ncic_curriculum_pdf(pdf_path.read_bytes(), subject_category_hint=cat)
        print(f"[{cat}] {pdf_path.name} -> {len(items)}개 성취기준 추출")
        all_extracted.extend(items)

    if not all_extracted:
        print("추출할 PDF 인자가 지정되지 않았습니다. --math / --science / --social 옵션을 확인하세요.")
        return 1

    res = merge_ncic_standards_to_catalog(all_extracted)
    stats = get_catalog_summary_stats()
    print(f"병합 완료: 신규 추가 {res['added']}개, 갱신 {res['updated']}개")
    print(f"현재 기준표 누적 현황: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
