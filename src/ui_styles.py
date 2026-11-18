from __future__ import annotations

import streamlit as st


def inject_responsive_css() -> None:
    """데스크톱·노트북·아이패드·갤럭시 탭·아이폰/안드로이드 전 기기 반응형 스타일 주입."""
    st.markdown(
        """
        <style>
        /* 전체 레이아웃 overflow 방지 및 반응형 패딩 */
        html, body, [data-testid="stAppViewContainer"] {
            max-width: 100%;
            overflow-x: hidden;
        }
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 3rem;
            padding-left: clamp(0.75rem, 2.5vw, 2.2rem);
            padding-right: clamp(0.75rem, 2.5vw, 2.2rem);
        }

        /* 상단 히어로 헤더 */
        .app-hero {
            background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 55%, #0EA5E9 100%);
            color: #FFFFFF;
            padding: clamp(1rem, 2.5vw, 1.6rem);
            border-radius: 16px;
            margin-bottom: 1.1rem;
            box-shadow: 0 8px 24px rgba(37, 99, 235, 0.16);
        }
        .app-hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: clamp(1.22rem, 2.6vw, 1.75rem);
            font-weight: 800;
            line-height: 1.3;
            color: #FFFFFF;
        }
        .app-hero p {
            margin: 0;
            font-size: clamp(0.84rem, 1.5vw, 0.96rem);
            color: #E0F2FE;
            line-height: 1.45;
        }

        /* 사이드바 메트릭 3자리 이상 숫자 잘림(12...) 방지 */
        [data-testid="stSidebar"] [data-testid="stMetric"] {
            padding: 0.35rem 0.25rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stMetricLabel"],
        [data-testid="stSidebar"] [data-testid="stMetricLabel"] p {
            font-size: 0.78rem !important;
            white-space: nowrap !important;
            overflow: visible !important;
        }
        [data-testid="stSidebar"] [data-testid="stMetricValue"],
        [data-testid="stSidebar"] [data-testid="stMetricValue"] > div {
            font-size: clamp(1.02rem, 1.3vw, 1.24rem) !important;
            line-height: 1.25 !important;
            white-space: nowrap !important;
            overflow: visible !important;
            text-overflow: unset !important;
        }

        /* 카드 UI */
        .info-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: clamp(0.85rem, 2vw, 1.2rem);
            margin-bottom: 0.9rem;
            color: #0F172A;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }

        /* 모드 B 상단 성적표 카드 */
        .score-report-card {
            background: linear-gradient(135deg, #EFF6FF 0%, #F8FAFC 60%, #F0FDF4 100%);
            border: 1.5px solid #93C5FD;
            border-radius: 14px;
            padding: 1rem 1.2rem;
            margin: 0.6rem 0 1.1rem 0;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.08);
            color: #0F172A;
        }
        .score-report-top {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: baseline;
            gap: 0.6rem;
            margin-bottom: 0.55rem;
        }
        .score-big {
            font-size: clamp(1.25rem, 2.2vw, 1.65rem);
            font-weight: 800;
            color: #1E3A8A;
        }
        .score-bar-track {
            width: 100%;
            height: 8px;
            background: #E2E8F0;
            border-radius: 999px;
            overflow: hidden;
            margin-bottom: 0.7rem;
        }
        .score-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #2563EB, #16A34A);
            border-radius: 999px;
        }
        .score-ox-row {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        .score-ox-pill {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
        }
        .score-ox-ok { background: #DCFCE7; color: #166534; border: 1px solid #86EFAC; }
        .score-ox-no { background: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; }

        /* 모던 학습지 스타일 문항 카드 */
        .q-badge-row {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 8px;
            margin-bottom: 0.35rem;
        }
        .q-num-pill {
            display: inline-block;
            background: linear-gradient(135deg, #1E3A8A, #2563EB);
            color: #FFFFFF;
            font-size: 0.9rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            padding: 0.25rem 0.72rem;
            border-radius: 8px;
        }
        .rewritten-card {
            background: #FFFFFF;
            border: 1.5px solid #CBD5E1;
            border-top: 4px solid #2563EB;
            border-radius: 12px;
            padding: clamp(1rem, 2.2vw, 1.45rem);
            margin: 0.7rem 0;
            color: #0F172A;
            line-height: 1.8;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
        }
        .rewritten-stem {
            font-size: clamp(0.96rem, 1.45vw, 1.05rem);
            line-height: 1.8;
            color: #0F172A;
            white-space: pre-wrap;
            word-break: keep-all;
            margin-bottom: 0.7rem;
        }
        .data-box {
            background: #F8FAFC;
            border: 1px solid #CBD5E1;
            border-left: 4px solid #475569;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin: 0.75rem 0;
            font-size: clamp(0.93rem, 1.35vw, 1rem);
            line-height: 1.75;
            color: #1E293B;
        }
        .data-box-title {
            font-size: 0.8rem;
            font-weight: 800;
            color: #334155;
            letter-spacing: 0.06em;
            margin-bottom: 0.35rem;
        }
        .bogi-box {
            position: relative;
            background: #FFFFFF;
            border: 1.5px solid #334155;
            border-radius: 8px;
            padding: 1.05rem 1rem 0.75rem 1rem;
            margin: 1rem 0 0.85rem 0;
            font-size: clamp(0.94rem, 1.4vw, 1.02rem);
            line-height: 1.75;
            color: #0F172A;
        }
        .bogi-title {
            position: absolute;
            top: -0.72rem;
            left: 50%;
            transform: translateX(-50%);
            background: #FFFFFF;
            padding: 0 0.75rem;
            font-size: 0.86rem;
            font-weight: 800;
            letter-spacing: 0.22em;
            color: #0F172A;
            white-space: nowrap;
        }
        .bogi-item {
            display: grid;
            grid-template-columns: 1.7rem 1fr;
            gap: 0.35rem;
            align-items: start;
            margin-bottom: 0.38rem;
            word-break: keep-all;
        }
        .bogi-item:last-child { margin-bottom: 0; }
        .bogi-marker { font-weight: 800; color: #1E3A8A; }
        .rewritten-choices-wrap {
            margin-top: 0.85rem;
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
        }
        .rewritten-choice {
            padding: 0.55rem 0.85rem;
            background: #F8FAFC;
            border: 1px solid #CBD5E1;
            border-radius: 9px;
            font-size: clamp(0.92rem, 1.38vw, 1rem);
            line-height: 1.65;
            color: #1E293B;
            transition: border-color 0.15s ease, background-color 0.15s ease;
        }
        .rewritten-choice:hover {
            border-color: #3B82F6;
            background: #EFF6FF;
        }
        .visual-term-note {
            background: #FFFBEB;
            border: 1px solid #FDE68A;
            border-left: 4px solid #D97706;
            border-radius: 8px;
            padding: 0.55rem 0.85rem;
            margin: 0.45rem 0 0.7rem 0;
            font-size: 0.88rem;
            font-weight: 600;
            color: #92400E;
        }
        .omr-row .stButton > button {
            border-radius: 999px !important;
            border: 1.5px solid #CBD5E1 !important;
            font-weight: 700 !important;
        }
        .omr-row .stButton > button[kind="primary"] {
            border-color: #1D4ED8 !important;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25) !important;
        }

        /* 배지 & 칩 */
        .chip-container {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin: 0.4rem 0 0.7rem 0;
        }
        .chip {
            display: inline-block;
            padding: 0.22rem 0.62rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            line-height: 1.35;
        }
        .chip-blue { background: #DBEAFE; color: #1E40AF; border: 1px solid #BFDBFE; }
        .chip-green { background: #DCFCE7; color: #166534; border: 1px solid #BBF7D0; }
        .chip-amber { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
        .chip-purple { background: #EDE9FE; color: #5B21B6; border: 1px solid #DDD6FE; }
        .chip-red { background: #FEE2E2; color: #991B1B; border: 1px solid #FECACA; }

        /* 회원 프로필 카드 & 등급 배지 */
        .account-card {
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-radius: 12px;
            padding: 0.75rem 0.85rem;
            margin-bottom: 0.7rem;
            color: #0F172A;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        }
        .tier-badge {
            display: inline-block;
            padding: 0.18rem 0.58rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.03em;
        }
        .tier-FREE { background: #F1F5F9; color: #334155; border: 1px solid #CBD5E1; }
        .tier-CUSTOM { background: #DCFCE7; color: #166534; border: 1px solid #86EFAC; }
        .tier-PRO { background: #DBEAFE; color: #1D4ED8; border: 1px solid #93C5FD; }
        .tier-ULTRA {
            background: linear-gradient(135deg, #F59E0B 0%, #EA580C 100%);
            color: #FFFFFF;
            border: none;
        }
        .tier-ADMIN {
            background: linear-gradient(135deg, #7C3AED 0%, #2563EB 100%);
            color: #FFFFFF;
            border: none;
        }

        /* 이미지 반응형 */
        img {
            max-width: 100% !important;
            height: auto !important;
            border-radius: 8px;
        }

        /* 터치 친화적 버튼 (최소 44px 높이) */
        .stButton > button, .stDownloadButton > button {
            min-height: 44px;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.45rem 0.9rem;
            transition: all 0.15s ease;
        }

        /* 탭 바 모바일 가로 스크롤 대응 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            overflow-x: auto;
            flex-wrap: nowrap;
            scrollbar-width: thin;
        }
        .stTabs [data-baseweb="tab"] {
            min-height: 44px;
            padding: 0.45rem 0.95rem;
            font-weight: 700;
            white-space: nowrap;
            border-radius: 10px 10px 0 0;
        }

        /* 모바일/소형 태블릿(<=768px) 최적화 */
        @media (max-width: 768px) {
            .block-container {
                padding-top: 0.75rem;
                padding-left: 0.65rem;
                padding-right: 0.65rem;
            }
            /* 다중 컬럼이 모바일에서 너무 좁게 찌그러지지 않도록 최소 폭 확보 */
            [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
                gap: 0.55rem !important;
            }
            [data-testid="column"] {
                min-width: min(100%, 280px) !important;
                flex: 1 1 280px !important;
            }
            /* 1~5 객관식 버튼용 가로 버튼 그룹은 모바일에서도 한 줄 5개 유지 가능하게 예외 허용 */
            .omr-row [data-testid="column"] {
                min-width: 48px !important;
                flex: 1 1 48px !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_chips(items: list[str], variant: str = "blue", limit: int = 80) -> None:
    if not items:
        return
    shown = items[:limit]
    html_chips = "".join(
        f"<span class='chip chip-{variant}'>{_html_escape(str(item))}</span>"
        for item in shown
    )
    if len(items) > limit:
        html_chips += f"<span class='chip chip-amber'>+{len(items) - limit}개 더</span>"
    st.markdown(f"<div class='chip-container'>{html_chips}</div>", unsafe_allow_html=True)


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
