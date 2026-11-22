from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36"
)


@dataclass
class EbsiFetchResult:
    success: bool
    message: str
    problem_pdf_bytes: bytes | None = None
    answer_pdf_bytes: bytes | None = None
    problem_filename: str = ""
    answer_filename: str = ""
    discovered_links: list[str] = field(default_factory=list)


def _is_pdf_bytes(data: bytes) -> bool:
    return data.startswith(b"%PDF")


def _looks_like_login_or_block(status_code: int, final_url: str, html_text: str) -> bool:
    if status_code in (401, 403, 429):
        return True
    lower_url = final_url.lower()
    if any(tok in lower_url for tok in ("login", "sso", "auth", "member", "captcha")):
        return True
    block_markers = (
        "로그인이 필요",
        "로그인 후 이용",
        "회원 로그인",
        "접근 권한이 없",
        "자동입력 방지",
        "보안 문자",
        "비정상적인 접근",
    )
    return any(m in html_text for m in block_markers)


def attempt_ebsi_pdf_fetch(url: str, timeout_sec: int = 15) -> EbsiFetchResult:
    """
    EBSi 기출 URL에서 공개 PDF 링크 추출 및 다운로드를 시도한다.
    로그인·보안 차단 시 우회하지 않고 즉시 수동 업로드 모드로 안내한다.
    """
    cleaned = url.strip()
    parsed = urlparse(cleaned)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return EbsiFetchResult(
            success=False,
            message="올바른 http/https URL을 입력해 주세요. 바로 아래 [수동 업로드]를 이용하시면 가장 빠르고 정확합니다.",
        )

    import requests
    from bs4 import BeautifulSoup

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.8",
            "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
        }
    )

    try:
        resp = session.get(cleaned, timeout=timeout_sec, allow_redirects=True)
    except requests.RequestException as exc:
        return EbsiFetchResult(
            success=False,
            message=f"EBSi 페이지 접속 중 네트워크/보안 오류가 발생했습니다({exc}). 우측/하단 [수동 업로드]로 PDF를 직접 올려주세요.",
        )

    # 1) URL 자체가 직접 PDF인 경우
    if _is_pdf_bytes(resp.content):
        fname = parsed.path.split("/")[-1] or "ebsi_exam.pdf"
        if not fname.lower().endswith(".pdf"):
            fname += ".pdf"
        return EbsiFetchResult(
            success=True,
            message="입력하신 링크에서 공개 문제지 PDF를 직접 다운로드했습니다.",
            problem_pdf_bytes=resp.content,
            problem_filename=fname,
        )

    html = resp.text or ""
    if _looks_like_login_or_block(resp.status_code, resp.url, html):
        return EbsiFetchResult(
            success=False,
            message="EBSi가 로그인 인증 또는 보안 정책으로 자동 수집을 차단했습니다(HTTP "
            f"{resp.status_code}). 보안 우회 없이 즉시 [수동 업로드] 탭에서 다운로드한 문제지/해설지 PDF를 올려주세요.",
        )

    # 2) HTML 내부에서 공개 PDF 링크 탐색
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[tuple[str, str]] = []  # (url, label)

    for tag in soup.find_all(["a", "button"]):
        href = (tag.get("href") or "").strip()
        onclick = (tag.get("onclick") or "").strip()
        label = tag.get_text(" ", strip=True)
        for raw_attr in (href, onclick):
            if not raw_attr:
                continue
            for m in re.findall(r"https?://[^\s'\"<>]+\.pdf(?:\?[^\s'\"<>]*)?", raw_attr, re.I):
                candidates.append((m, label))
            if ".pdf" in raw_attr.lower() and not raw_attr.lower().startswith("javascript:"):
                candidates.append((urljoin(resp.url, raw_attr), label))

    # 중복 제거
    seen: set[str] = set()
    unique_candidates: list[tuple[str, str]] = []
    for cand_url, lbl in candidates:
        if cand_url not in seen:
            seen.add(cand_url)
            unique_candidates.append((cand_url, lbl))

    if not unique_candidates:
        return EbsiFetchResult(
            success=False,
            message="EBSi 페이지는 JavaScript 동적 다운로드 또는 회원 인증 방식으로 PDF를 제공합니다. "
            "브라우저에서 직접 PDF를 받으신 뒤 아래 [수동 업로드]를 이용해 주세요.",
        )

    prob_bytes: bytes | None = None
    ans_bytes: bytes | None = None
    prob_name = "ebsi_problem.pdf"
    ans_name = "ebsi_answer.pdf"

    for cand_url, lbl in unique_candidates[:6]:
        try:
            dl = session.get(cand_url, timeout=timeout_sec, allow_redirects=True)
            if not _is_pdf_bytes(dl.content):
                continue
            is_answer = any(k in (lbl + cand_url).lower() for k in ("해설", "정답", "해답", "ans", "sol"))
            fname = urlparse(cand_url).path.split("/")[-1] or ("ebsi_answer.pdf" if is_answer else "ebsi_problem.pdf")
            if is_answer and ans_bytes is None:
                ans_bytes = dl.content
                ans_name = fname
            elif prob_bytes is None:
                prob_bytes = dl.content
                prob_name = fname
        except requests.RequestException:
            continue

    if prob_bytes is None and ans_bytes is None:
        return EbsiFetchResult(
            success=False,
            message="PDF 링크가 감지되었으나 EBSi 로그인/보안 세션이 필요하여 다운로드가 차단되었습니다. "
            "즉시 [수동 업로드]로 PDF를 등록해 주세요.",
            discovered_links=[u for u, _ in unique_candidates],
        )

    return EbsiFetchResult(
        success=True,
        message="공개 PDF 추출에 성공했습니다.",
        problem_pdf_bytes=prob_bytes,
        answer_pdf_bytes=ans_bytes,
        problem_filename=prob_name,
        answer_filename=ans_name,
        discovered_links=[u for u, _ in unique_candidates],
    )
