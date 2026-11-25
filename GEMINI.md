# MogiMogi (2022 개정 교육과정 맞춤형 기출 플랫폼) 프로젝트 규칙 & 영구 컨텍스트

> **문서 목적**: Antigravity에서 새 대화를 생성할 때마다 별도 탐색 없이 즉시 전체 프로젝트 구조, 도메인 규칙, 데이터 스키마, 배포 환경을 파악하고 일관되게 작업하기 위한 영구 프로젝트 지식 베이스입니다. (항상 24 KB 미만으로 유지하여 Rules 예산 내 전문 인라인 로드를 보장합니다.)

---

## 1. 프로젝트 Git 자동 동기화 & 맥락 최신화 규칙 (최우선 절대 규칙)

1. **사전 동기화 (`git pull`)**: 모든 코드 수정이나 파일 생성 요청을 처리하기 직전에, 반드시 터미널에서 `git pull origin main`을 먼저 실행해 최신 코드를 동기화한다.
2. **사후 자동 커밋 & 푸시 (`git push`)**: 코드 수정과 작업이 완료되면, 사용자에게 최종 완료 답변을 출력하기 직전에 반드시 터미널에서 아래 순서대로 실행해 GitHub에 업로드한다.
   - `git add .`
   - `git commit -m "AI 자동 업데이트"`
   - `git push origin main`
3. **충돌 안전장치**: 만약 `git pull`이나 `git push` 과정에서 충돌(Conflict)이나 에러가 발생하면 임의로 강제 푸시(`--force`)하지 말고 즉시 멈춘 뒤 사용자에게 상황을 보고한다.
4. **영구 맥락 최신화**: 새로운 모듈/탭/데이터 스키마/배포 설정이 추가되거나 아키텍처가 변경될 때마다 이 `GEMINI.md`의 해당 섹션도 함께 업데이트하여 다음 새 대화에서도 맥락이 100% 이어지도록 유지한다.
5. **Notion 업데이트 로그 자동 기록**: `git push origin main`이 성공하면, Notion MCP(`notion_notion` → `notion-create-pages`)를 호출해 **모기고사(MogiMogi) 업데이트 로그** 데이터베이스(`data_source_id`: `42072e5d-0b8f-81d6-8baa-000bbde62563`, URL: `https://app.notion.com/p/42072e5d0b8f815f9f16fe8ee6fd3270`)에 이번 업데이트 항목을 1건 자동 추가한다. (단, Notion MCP 연결 불가 시 에러만 알리고 푸시는 유지한다.)
   - **속성(properties)**: `"업데이트 제목"`(핵심 변경 요약 제목), `"date:날짜:start"`(`YYYY-MM-DD`, `"date:날짜:is_datetime": 0`), `"유형"`(`✨ 새 기능` / `⚡ 개선/최적화` / `🐛 버그 수정` / `🎨 UI/UX` / `🔧 인프라/설정` / `📝 문서` 중 1개), `"영향 영역"`(`["수학/사/과/한", "영어", "국어", "영단어", "실전 풀이", "기출 업로드/DB", "관리자/과금", "공통/인프라"]` 중 해당 항목 배열), `"한 줄 요약"`(일반 사용자도 이해할 쉬운 한 줄 요약), `"커밋"`(`git rev-parse --short HEAD` 해시).
   - **페이지 본문(content, 하이브리드 형식)**: `## 주요 변경 사항`(사용자 눈높이 불릿 요약) + `## 개발 상세`(변경된 주요 파일, 기술적 변경 내역, 커밋 해시).

---

## 2. 프로젝트 개요 (Project Overview)

- **서비스명**: 2022 개정 교육과정 맞춤형 기출 추출 & 상위 문제 개조 플랫폼 (`mogigosa-bunsuck`)
- **핵심 도메인 로직**:
  - **수학 · 사회 · 과학 · 한국사**: 개념서·교과서 PDF(또는 공용 개념서 DB)에 실제로 등장한 핵심 원리(`core_principles`)와 용어(`required_terms` / `basic_terms`)만 허용하고, 원리는 같지만 상위 학년·선택과목 전문 용어(`advanced_terms`)가 포함된 기출문제를 학생의 개념서 언어로 자동 개조(`RewrittenQuestion`)한다.
  - **영어**: 수능/모의고사 영어 독해 지문을 기반으로 11종 수능 유형(`ENGLISH_TRANSFORM_TYPES`)으로 변형 문제를 생성하며(`English Engine v3`: `src/english_v3/` — 원문 정화·문장 인덱싱 `[S0]~[Sn-1]` + 무주석 경량 좌표/델타 Pydantic 스키마 + `순서`/`삽입` Python 담화 응집도 0토큰 후보 조립 + 0토큰 시드 어휘 연동 + Gate 1 하드 규칙 검증 + 중·고난도 8종 정답 은닉 Blind Solver Gate 2 + `gemini-3.5-flash-lite` 내 `thinking_level` 차등 라우팅(`minimal`/`low`→`medium`) + 3문항 이상 Flex 50% 할인 & Batch API + `v3` 캐시 풀 격리), 고품질 문제·해설·영단어장 PDF를 제공한다. 또한 **영어 지문 분석지**(주제 · 논리 전개 흐름 · 공식 각주 · 이해용 시각 도식 · AI 미사용 문제지/해설지 PDF 순수 복붙 문장별 원문+해설 매칭, `analysis_version=5` / PDF `v7`, 기출 포인트 제거)를 생성·캐싱·PDF 출력할 수 있다.
  - **국어**: 독서·문학·화법과 작문·언어와 매체 세트 지문을 분석하고, 자유 유형 세트 문항 재구성, Pillow 기반 시각 구조도(`diagram_renderer.py`), 2단/분석본 PDF(`korean_pdf_generator.py`)를 생성한다.
- **기술 스택**: Python, Streamlit (`>=1.50.0`), PyMuPDF (`pymupdf`), Google GenAI SDK (`google-genai`), Pydantic v2 (`pydantic`), Pillow (`pillow`), Requests, BeautifulSoup4.

---

## 3. AI 모델 & 과금 / 인증 정책 (`src/config.py`, `src/auth.py`, `src/fx_rate.py`)

### 3.1 단일 AI 모델 원칙
- 전 기능(기출 태깅, 무정답 풀이, 개념서 단원/화이트리스트 추출, 문항 개조, 영어/국어 변형, 영단어 사전) 모두 **`gemini-3.5-flash-lite`** (`FLASH_LITE_MODEL`) 단일 모델로 통일되어 있다.
- `resolve_api_model_name()`은 모든 모델 요청을 `gemini-3.5-flash-lite`로 정규화한다.

### 3.2 토큰 단가 & 일일 환율 자동 갱신
- **공식 API 단가 (USD / 1M tokens)**: Input `$0.30` (`FLASH_LITE_INPUT_USD_PER_1M`), Cached Input `$0.03` (`FLASH_LITE_CACHED_INPUT_USD_PER_1M`), Output(사고 토큰 포함) `$2.50` (`FLASH_LITE_OUTPUT_USD_PER_1M`).
- **환율 동기화 (`src/fx_rate.py`)**: `app.py` 시작 시 `start_daily_midnight_fx_scheduler()`가 구동되며 매일 00:00 KST 및 온디맨드로 Google Cloud Billing Catalog API(`6F81-5844-456A`)를 1순위 조회하고, 폴백 환율 조회 결과를 `exam_db/fx_rate_cache.json`에 캐싱한다(기본 폴백 `DEFAULT_USD_TO_KRW = 1335.0`).

### 3.3 회원 등급(`TierType`) & 정액 크레딧(`Credit`) / 일일 방어 정책 (KST 매월 1일 초기화)
| 등급 | 표기 / 구독료 | 월간 기본 크레딧 | `0C` 무료 AI 일일 한도 | 공용 DB 신규 업로드 *(50MB 이하)* | 주요 권한 및 제한 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `FREE` | 무료 (0원) | `30 C` | 일 10회 *(채팅 5턴)* | 불가 (열람만) | 1회 배치 최대 5문항 (`FREE_MAX_BATCH_QUESTIONS = 5`) |
| `PRO` | PRO (9,900원/월) | `600 C` | 일 40회 *(채팅 5턴)* | 일 3회 (`0C`) | 배치 제한 해제 · 공용 DB 신규 업로드/불러오기 가능 |
| `ULTRA` | ULTRA (19,900원/월) | `2,400 C` | 일 100회 *(채팅 5턴)* | 일 8회 (`0C`) | PRO 대비 정확히 4배 크레딧 · 공용 DB 신규 업로드/불러오기 가능 |
| `ADMIN` | 최고 관리자 | 무제한 | 무제한 | 무제한 | 기본 관리자 `sibagetory10`, 공용 DB 수정·삭제·NCIC 관리 단독 권한 |

- **단일 서버 키 & `CUSTOM` 폐지**: 레거시 `CUSTOM`/`BYOK` 계정은 로드 시 `FREE`로 자동 이관되며 `custom_api_key`도 공백으로 파기된다.
- **정액 크레딧 단가 (신규 생성 · 캐시 풀 히트 동일)**: 영어 변형 `2C/문항`, 영어 분석지 `4C/지문`, 국어 세트 변형·분석지 각 `10C/세트`, 수·사·과·한 `개조 3C / 무개조 1C`, 개인 개념서 PDF 추출 `3C/회`, 개별 재변환 `영어 2C / 수·사·과·한 3C`. AI 해설·1:1 채팅·영단어 사전은 `0C` 무료(단, 시드/캐시 히트 외 신규 AI 호출 시 일일 무료 한도 차감).
- **원자적 선예치·환불 & 원장**: `acquire_credit_reservation` / `settle_credit_reservation`으로 동시 실행 차단 및 월간 기본 크레딧 우선 차감 후 `bonus_credits` 차감·미성공분 자동 환불을 수행하며 모든 변동은 `exam_db/credit_ledger.jsonl`에 기록된다.
- **무통장입금·송금 링크 결제 수명주기 (`src/auth/payments.py`)**: 사업자등록 전 단계 계좌이체/카카오·토스 링크 안내 → 회원 입금 확인 신청(`PENDING`, 중복 차단) → Discord 웹훅 실시간 알림 → 관리자 원클릭 승인(`APPROVED` · 당월 `0C` 리셋 · 동일 등급 `+30일` 합산 연장 / 타 등급 `승인일+30일`) 및 반려(`REJECTED`). 만료 시각(`tier_expires_at`) 도래 시 `get_effective_tier()` 0초 즉시 `FREE` 판정 + 60초·자정 자동 스윕(`maybe_sweep_expired_users`)으로 강등. 승인 건 환불(`REFUNDED`) 시 직전 구독 복원 및 승인 전후 소비량 역산(`refund_credit_debt` 상계)으로 크레딧 이중 수령을 원천 차단하며, 관리자 표 수동 승급(지인·테스터)은 무기한 유지 및 실매출 집계에서 제외한다.
- **인증 방식**: 로컬 ID/PW 로그인(PBKDF2-HMAC-SHA256 + 서명된 세션 토큰 유지) 및 Google OAuth 2.0 웹 로그인(`src/auth/oauth.py`) 동시 지원.

---

## 4. 디렉터리 및 모듈 아키텍처 맵 (File & Module Map)

### 4.1 진입점 & UI 레이어 (`app.py`, `src/ui/`)
- **`app.py`**: 페이지 설정(`layout="wide"`), CSS 주입(`inject_responsive_css`), `init_session_defaults()`, `get_cached_storage(local_root="./exam_db")`, 환율 스케줄러 시작, 사이드바 인증 후 `create_lazy_tabs()`로 **열린 탭만 지연 임포트/렌더링**.
- **`src/ui/context.py`**: 탭 간 공유 컨텍스트 `AppContext` (`TYPE_CHECKING` 기반 경량 타입 임포트).
- **`src/ui/common.py`**: `init_session_defaults`, `get_cached_storage`, `create_lazy_tabs` / `create_lazy_expander`, 경량 토큰 사용량 추적(`src.gemini.usage`), 시험지 자동 저장(`_auto_save_current_exam`), 답안 정규화(`normalize_answer`).
- **`src/ui/auth_sidebar.py` & `src/ui/payment_modal.py`**: 로그인/회원가입/Google OAuth(`src.auth.oauth`), 멤버십 만료일·크레딧 원장 표시, `@st.dialog` 결제 신청·송금 안내 모달, `get_db_summary_counts()` O(1) DB 요약, 게스트 랜딩.
- **`src/ui/tab1_upload.py`, `tab1_pairing.py`, `tab1_tagging.py`, `tab1_batch_queue.py`, `tab1_db_explorer.py` [탭 1]**: 모의고사 PDF 수집(단일/대기열/EBSi), AI 표지·해설지 매칭, 문항 크롭(`CroppedQuestion`), 2022 원리·용어 분해 태깅, 과목별 기출 DB 탐색/수정/내보내기.
- **`src/ui/tab2_transform.py` [탭 2 메인]**: 기준 개념서 범위 설정(공용 개념서 DB / 직접 PDF) → 화이트리스트 추출 → 기출 매칭 & 상위 용어 개조, 영어/국어 전용 변형 섹션 지연 분기.
- **`src/ui/tab2_english.py` & `src/ui/tab2_english_analysis.py` [탭 2 - 영어]**: 11종 수능 독해 유형 변형 배치 실행 및 **영어 정밀 지문 분석지 PDF 생성/미리보기**.
- **`src/ui/tab2_korean.py` [탭 2 - 국어]**: 독서·문학·화작·언매 지문 세트 선택, 지문 심층 분석, 시각 구조도 생성, 세트 전체 문항 재구성 및 PDF 생성/병합.
- **`src/ui/tab2_quiz.py` [탭 2 - 실전 풀이]**: `@st.fragment` 기반 웹 바로 풀기, 자동 채점 성적표, 개념서 근거 기반 AI 해설 & 채팅, 문항별 개별 재변환.
- **`src/ui/tab3_concept.py` [탭 3]**: 공용 개념서·교과서 DB 구축 및 2022 개정 교육과정 공식 기준표(NCIC 고시 PDF / 교육부 원문 동기화) 관리.
- **`src/ui/tab4_dictionary.py` & `src/ui/tab4_vocab_quiz.py` [탭 4]**: 수능 영단어 사전(`data/csat_vocab_seed.json` + AI 검색 캐시), 기출 핵심 어휘 연동, 나만의 단어장 & PDF, 객관식/플래시카드 퀴즈.
- **`src/ui/my_exams_tab.py` [나의 시험지]**: 계정별·과목별 자동 저장된 시험지 목록 조회, PDF 다운로드, 웹 재풀이 로드.
- **`src/ui/admin_tab.py` & `src/ui/admin_pools.py` [관리자]**: 입금 승인 대기열(원클릭 승인·반려·환불 회수)·입금 계좌/송금 링크/Discord 웹훅 설정, 입금 승인 실매출 기반 원가율 대시보드, 회원 등급·만료일 관리, 서버 디스크/Git/풀 관리.

### 4.2 Gemini AI 도메인 서비스 (`src/gemini/` & 파사드)
- **`src/gemini/usage.py` & `src/gemini/core.py`**: 경량 토큰 추적(`bind_usage_tracker`, `submit_with_usage_context`, `ForbiddenTermRemainingError`)과 GenAI 클라이언트·재시도·폴백·비전 최적화·구조화 파싱 분리.
- **`src/gemini/exam_tagging.py`**: 시험지 헤더 메타데이터 추출, 문제지-해설지 AI 자동 페어링, 스캔본 문항 박스 감지, 과목별 배치 문항 태깅.
- **`src/gemini/concept_book.py`**: 개념서 표지/목차/본문 단원 추출, 개념 화이트리스트(`ConceptWhitelist`) 추출, 문항-화이트리스트 적합성 매칭.
- **`src/gemini/rewrite.py`**: 금지 상위 용어 제거 문항 개조(`rewrite_question_for_pdf`), 개념서 근거 해설, 후속 질의응답.
- **`src/english_v3/` (`routing.py`, `schemas.py`, `preprocessor.py`, `discourse.py`, `transport.py`, `assembler.py`, `rules_gate.py`, `blind_validator.py`, `pipeline.py`, `batch_runner.py`)**: 백지 재설계 수능 영어 11종 변형 엔진 v3 (초경량 좌표/델타 스키마, `순서`/`삽입` 담화 0토큰 조립 + 단일 Blind Solver 통합 검수, 파이썬 하드규칙 Gate 1, 8종 Blind Solver Gate 2, `thinking_level` 차등 라우팅, Standard/Flex/Batch 50% 할인).
- **`src/gemini/english_tagging.py`, `english_transform.py`, `english.py`(지연 파사드)**: 영어 문항 태깅 및 정밀 지문 분석지 PDF(`generate_english_passage_analysis`) 전용; 변형문제·해설·채팅 export는 `src/english_v3/pipeline.py`로 연결.
- **`src/gemini/korean.py`**: 국어 지문 노이즈 제거, 세트 지문 동기화, 문학 작품 웹 브리프 조회, 지문 심층 분석·교열, 시각 다이어그램, 세트 문항 재구성.
- **`src/gemini/dictionary.py`**: 영단어 Gemini 사전 조회(`lookup_english_word_with_gemini`).
- **하위 호환 지연 파사드**: `src/gemini/__init__.py` (`_SUBMODULE_EXPORTS`), `src/gemini_service.py`, `src/korean_gemini_service.py`.

### 4.3 코어 스키마 · 인증 · 저장소 · PDF · 교육과정 엔진 (`src/`)
- **`src/models/` (`question.py`, `concept.py`, `korean.py`, `english.py`, `user.py`, `payment.py`, `vocab.py`, `__init__.py`)**: PEP 562 지연 export 기반 Pydantic v2 스키마 패키지.
- **`src/auth/` (`tokens.py`, `oauth.py`, `accounts.py`, `quota.py`, `payments.py`, `__init__.py`)**: PBKDF2 인증, 서명 세션 토큰, Google OAuth 2.0, 30일 유료 만료·환불 부채 상계·결제 승인 서비스, 등급·크레딧 쿼터 관리.
- **`src/storage/` (`base.py`, `exam_store.py`, `user_exams_store.py`, `payment_store.py`, `english_pool_store.py`, `analysis_pool_store.py`, `vocab_store.py`, `__init__.py`)**: `StorageManager` — `threading.RLock` + 원자적 파일 교체(`_atomic_write_text`/`_bytes`), `mtime_ns`/`size` 캐시, 읽기 전용 `load_db_readonly()` & O(1) `get_db_summary_counts()`.
- **`src/curriculum_2022/` (`catalog.py`, `importers.py`, `grounding.py`, `__init__.py`)**: 2022 개정 교육과정 편제표·NCIC 성취기준 로드·파싱·검증.
- **`src/pdf_parser/` (`common.py`, `metadata_and_pairing.py`, `concept_slicer.py`, `exam_cropper.py`, `english_source_extractor.py`, `__init__.py`)**: PyMuPDF 기반 문항 영역 크롭, 2단 레이아웃 파싱, 개념서 슬라이싱, 해설지 매칭, 영어 문제지/해설지 `[해석]` 무손실 추출 및 문장별 1:1 DP 정렬(AI 미사용).
- **`src/pdf/` (`styles.py`, `common.py`, `poetry_renderer.py`, `korean_analysis_pdf.py`, `english_analysis_pdf.py`, `exam_pdf.py`, `vocab_pdf.py`, `__init__.py`)**: 맞춤형 2단 시험지, 국어·영어 지문 분석지, 영단어장 PDF 렌더링.
- **`src/diagram_renderer/` (`primitives.py`, `academic_charts.py`, `literature_charts.py`, `card_layouts.py`, `__init__.py`)**: Pillow 기반 국어·영어 지문 시각 구조도 PNG 렌더링.
- **`src/text_formatter.py`, `src/ui_styles.py`, `src/exporters.py`, `src/ebsi_fetcher.py`**: 화학/수식 첨자 및 영어/국어 블록 포매팅, 반응형 CSS, JSONL/HTML/MD 내보내기, EBSi 연동.

---

## 5. 영구 데이터 저장소 구조 (`./data`, `./exam_db`, `./concept_db`)

- **Git 추적 정적 데이터 (`data/`)**:
  - `data/csat_vocab_seed.json`: 수능 필수 영단어 내장 시드 DB
  - `data/curriculum_2022.json` & `data/curriculum_2022_moe_seed.json`: 2022 개정 교육과정 편제표 및 교육부 고시 제2022-33호 시드
- **런타임 영구 DB (`.gitignore` 제외 대상, 서버 디스크 보존)**:
  - `exam_db/db.json` 및 `exam_db/과목별_기출DB/`: 전체 기출 문항 메타데이터 및 과목별 정리본
  - `exam_db/pdfs/`, `exam_db/crops/`: 업로드된 원본 PDF 및 크롭 문항/지문 PNG
  - `exam_db/users.json`, `exam_db/credit_ledger.jsonl`, `exam_db/payment_settings.json`, `exam_db/payment_orders.json`: 회원 계정, 등급·만료일·환불부채·크레딧·원가 통계, 크레딧 원장 및 입금 계좌/결제 주문 DB
  - `exam_db/fx_rate_cache.json`: 일일 USD→KRW 환율 캐시
  - `exam_db/english_question_pool.json`, `english_analysis_pool.json`, `english_analysis_pdfs/`, `user_english_history.json`, `english_dict_cache.json`, `user_vocab/`
  - `exam_db/korean_analysis_pool.json`, `korean_analysis_pdfs/`
  - `exam_db/saved_exams/`: 사용자별 자동 저장된 맞춤형 시험지 및 PDF
  - `exam_db/ncic_standards.json`: 런타임에 추가된 NCIC 성취기준 (자동 업데이트 시 `data/curriculum_2022.json` 충돌 방지를 위해 우선 이관됨)
  - `concept_db/concept_books.json`: 공용 개념서·교과서 단원 및 화이트리스트 DB

---

## 6. 개발 및 코딩 핵심 원칙 (Invariants & Pitfalls)

1. **원자적 저장 & 스레드 락 엄수**: 영구 JSON/파일 쓰기 시 직접 `open(..., "w")`로 덮어쓰지 말고 반드시 `StorageManager`의 `self._lock` (`RLock`)과 `_atomic_write_text` / `_atomic_write_bytes` 패턴을 사용한다.
2. **워커 스레드 토큰 과금 전파**: `ThreadPoolExecutor` 등 병렬 스레드에서 Gemini API를 호출할 때는 반드시 `submit_with_usage_context`를 사용해 부모 스레드의 `_usage_scope` 토큰 레코더가 누락 없이 전파되도록 한다.
3. **영어 · 국어 세트 그룹 규격**:
   - **영어 (`normalize_english_group_id`)**: 실제로 지문을 공유하는 `41~42`, `43~45`만 세트 그룹으로 인정한다 (`31~34` 등은 공통 발문 헤더일 뿐 개별 지문이므로 빈 문자열 처리).
   - **국어 (`normalize_korean_group_id`)**: `[4~9]` 등을 `4~9`로 정규화하며 세트 내 문항들은 동일 지문을 공유한다(`_sync_korean_group_passages`).
4. **하위 호환 파사드 동기화**: `src/gemini/` 하위 모듈에 심볼을 추가·이름 변경할 때는 `src/gemini/__init__.py`의 `_SUBMODULE_EXPORTS` 및 `src/gemini_service.py` / `src/korean_gemini_service.py` 파사드 export도 함께 맞춘다.
5. **실서버 자동 배포 주의**: Oracle Cloud Ubuntu 24.04 단일 프로세스 `mogigosa.service` + 1분 주기 `mogigosa-autoupdate.timer` (`scripts/oracle_auto_update.sh`)가 `origin/main`의 새 커밋을 감지해 자동 `git pull --ff-only` 및 서비스 재시작을 수행한다. `main` 푸시 전 반드시 문법·임포트 오류가 없는지 확인한다.
