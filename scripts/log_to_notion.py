"""
모기고사(MogiMogi) 업데이트 로그 Notion 데이터베이스 자동 기록 스크립트.

사용법:
  python scripts/log_to_notion.py \
    --title "업데이트 제목" \
    --type "🎨 UI/UX" \
    --areas "영어" \
    --summary "한 줄 요약" \
    --commit "4b7d65a" \
    --details-file notes.md

필수 환경변수:
  NOTION_API_KEY: Notion Internal Integration Token (ntn_... 또는 secret_...)
  (프로젝트 루트의 .env 파일에 NOTION_API_KEY=... 형태로 저장해 두면 자동 로드됩니다.)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import requests

DATABASE_ID = "42072e5d0b8f815f9f16fe8ee6fd3270"
NOTION_API_URL = "https://api.notion.com/v1/pages"
NOTION_VERSION = "2022-06-28"


def _load_dotenv_if_present() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _markdown_to_notion_blocks(md_text: str) -> list[dict]:
    blocks: list[dict] = []
    for raw_line in (md_text or "").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        stripped = line.lstrip()
        if stripped.startswith("## "):
            blocks.append(
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"type": "text", "text": {"content": stripped[3:].strip()[:2000]}}]
                    },
                }
            )
        elif stripped.startswith("- "):
            blocks.append(
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": stripped[2:].strip()[:2000]}}]
                    },
                }
            )
        else:
            blocks.append(
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": stripped[:2000]}}]
                    },
                }
            )
    return blocks[:100]


def create_update_log(
    *,
    title: str,
    date_str: str,
    log_type: str,
    areas: list[str],
    summary: str,
    commit_hash: str,
    body_markdown: str,
) -> dict:
    _load_dotenv_if_present()
    api_key = os.environ.get("NOTION_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "NOTION_API_KEY가 설정되지 않았습니다. 프로젝트 루트의 .env 파일에 NOTION_API_KEY=... 를 추가해 주세요."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION,
    }
    payload = {
        "parent": {"database_id": DATABASE_ID},
        "properties": {
            "업데이트 제목": {"title": [{"text": {"content": title}}]},
            "날짜": {"date": {"start": date_str}},
            "유형": {"select": {"name": log_type}},
            "영향 영역": {"multi_select": [{"name": a} for a in areas if a.strip()]},
            "한 줄 요약": {"rich_text": [{"text": {"content": summary}}]},
            "커밋": {"rich_text": [{"text": {"content": commit_hash}}]},
        },
        "children": _markdown_to_notion_blocks(body_markdown),
    }

    resp = requests.post(NOTION_API_URL, headers=headers, json=payload, timeout=20)
    if resp.status_code >= 400:
        raise RuntimeError(f"Notion API 오류 ({resp.status_code}): {resp.text}")
    return resp.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="모기고사 Notion 업데이트 로그 기록")
    parser.add_argument("--title", required=True, help="업데이트 제목")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="YYYY-MM-DD")
    parser.add_argument("--type", required=True, dest="log_type", help="유형 (예: 🎨 UI/UX)")
    parser.add_argument("--areas", nargs="+", required=True, help="영향 영역 목록 (예: 영어)")
    parser.add_argument("--summary", required=True, help="한 줄 요약")
    parser.add_argument("--commit", required=True, help="커밋 해시")
    parser.add_argument("--details", default="", help="본문 마크다운")
    parser.add_argument("--details-file", default="", help="본문 마크다운 파일 경로")
    args = parser.parse_args()

    body = args.details
    if args.details_file:
        body = Path(args.details_file).read_text(encoding="utf-8")

    result = create_update_log(
        title=args.title,
        date_str=args.date,
        log_type=args.log_type,
        areas=args.areas,
        summary=args.summary,
        commit_hash=args.commit,
        body_markdown=body,
    )
    print(f"Notion 업데이트 로그 추가 완료: {result.get('url', result.get('id', ''))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
