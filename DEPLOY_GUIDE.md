# ☁️ 24시간 클라우드 무료 배포 가이드 (GitHub + Streamlit Community Cloud)

내 PC가 꺼져 있어도 **아이폰, 아이패드, 갤럭시 탭, 노트북, PC 어디서나** 24시간 접속 가능한 나만의 **2022 개정 교육과정 맞춤형 기출 추출 & 문제 개조 플랫폼**을 3분 만에 배포하는 방법입니다.

---

## 📦 1단계: 사전 준비 (API 키 & 구글 드라이브 5TB 연동)

### 1) Gemini API 키 발급
1. [Google AI Studio](https://aistudio.google.com/apikey) 접속 → Google 계정 로그인
2. **Create API key** 클릭 → 발급된 `AIzaSy...` 키 복사

### 2) Google Drive 영구 저장소 폴더 & 서비스 계정(Service Account) 설정
Streamlit Community Cloud 서버는 재부팅 시 로컬 파일이 초기화될 수 있으므로, **Google Drive(5TB)에 `db.json`, 원본 PDF, 문항 PNG를 영구 저장**하도록 연결합니다.

1. **Google Cloud Console** ([console.cloud.google.com](https://console.cloud.google.com)) 접속 → 새 프로젝트 생성
2. 좌측 메뉴 **API 및 서비스 → 라이브러리** → `Google Drive API` 검색 → **사용** 클릭
3. **IAM 및 관리자 → 서비스 계정** → **서비스 계정 만들기**
   - 이름 입력(예: `exam-platform-drive`) → 생성 완료
   - 생성된 서비스 계정 클릭 → **키(Keys)** 탭 → **키 추가 → 새 키 만들기 → JSON** 선택하여 `.json` 파일 다운로드
   - *(주의: 이 JSON 파일은 `.gitignore`에 의해 GitHub에 올라가지 않으므로 안전하게 보관하세요.)*
4. **Google Drive 폴더 공유 설정**:
   - 내 드라이브 또는 5TB 공유 드라이브에 저장용 폴더 생성 (예: `Exam_Platform_DB`)
   - 브라우저 주소창의 폴더 ID 복사: `https://drive.google.com/drive/folders/<여기가_FOLDER_ID>`
   - 폴더 우클릭 → **공유** → 서비스 계정 이메일(`xxx@yyy.iam.gserviceaccount.com`)을 **편집자(공유 드라이브인 경우 콘텐츠 관리자)** 권한으로 추가

---

## 🚀 2단계: GitHub 업로드 (보안 파일 자동 제외)

프로젝트에 포함된 `.gitignore` 파일이 `.venv/`, `.streamlit/secrets.toml`, `exam_db/`, `*.json` 등 민감·임시 파일을 자동으로 제외하므로 안심하고 업로드할 수 있습니다.

```bash
git init
git add .
git commit -m "Deploy 2022 curriculum customized exam platform"
git branch -M main
git remote add origin https://github.com/<내_아이디>/<저장소_이름>.git
git push -u origin main
```

> **체크리스트**: GitHub 저장소에 아래 핵심 파일이 올라갔는지 확인하세요.
> - `app.py`
> - `src/` 폴더 전체
> - `requirements.txt`
> - `packages.txt` (클라우드 리눅스 한글 폰트 지원용)
> - `.streamlit/config.toml`
> - `.streamlit/secrets.toml.example`
> - **절대 올라가면 안 되는 파일**: `.streamlit/secrets.toml`, `*.json`, `.venv/`

---

## ⚡ 3단계: Streamlit Community Cloud 3분 무료 배포

1. [share.streamlit.io](https://share.streamlit.io/) 접속 → GitHub 계정으로 로그인
2. 우측 상단 **Create app** (또는 **New app**) 클릭
3. 설정 입력:
   - **Repository**: `<내_아이디>/<저장소_이름>`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. 하단 **Advanced settings...** 클릭 → **Secrets** 입력창에 `.streamlit/secrets.toml.example` 형식을 참고해 실제 키 입력:

```toml
GEMINI_API_KEY = "AIzaSy..."
GOOGLE_DRIVE_FOLDER_ID = "1AbCDefGHiJkLmNoPqRsTuVwXyZ"
ADMIN_USERNAME = "sibagetory10"

GCP_SERVICE_ACCOUNT = '''
{
  "type": "service_account",
  "project_id": "...",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "...@....iam.gserviceaccount.com",
  "client_id": "...",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "..."
}
'''
```

5. **Deploy!** 클릭 → 약 1~2분 빌드 후 24시간 고유 URL(`https://<앱이름>.streamlit.app`) 발급 완료!

---

## 📱 4단계: 아이폰·아이패드·갤럭시 탭에서 앱처럼 사용하기

1. 발급된 `https://<앱이름>.streamlit.app` 주소로 모바일/태블릿 브라우저 접속
2. **아이폰/아이패드 (Safari)**: 공유 버튼(□↑) → **홈 화면에 추가**
3. **갤럭시 탭/안드로이드 (Chrome/삼성 인터넷)**: 메뉴(⋮) → **현재 페이지 추가 → 홈 화면**
4. 앱 시작 시 Google Drive와 자동 동기화되므로 PC에서 수집한 기출 DB를 태블릿에서 바로 풀고 AI 자동 해설을 받을 수 있습니다!

---

## 🔧 자주 묻는 질문 (트러블슈팅)

- **Q. Google Drive 연동이 안 되고 "로컬 폴백"으로 뜹니다.**
  - 서비스 계정 이메일(`client_email`)을 대상 Google Drive 폴더(또는 공유 드라이브)에 **편집자 / 콘텐츠 관리자**로 초대했는지 확인하세요.
  - GCP 프로젝트에서 **Google Drive API**가 활성화되어 있는지 확인하세요.
- **Q. 장시간 미사용 후 접속이 느립니다.**
  - Streamlit Community Cloud 무료 티어는 일정 시간 접속이 없으면 슬립 모드로 들어가며, 다시 접속하면 약 10~20초 내에 자동 기상하여 Google Drive 데이터를 다시 동기화합니다.
- **Q. 개인 API 키가 없으면 사용할 수 없나요?**
  - 회원가입 시 기본 `FREE`(무료 회원) 등급으로도 **기출 PDF 업로드를 무제한**으로 이용할 수 있으며, 기본 추론은 **Gemini 3.5 Flash-Lite**로 실행되고 **기출 3회 업로드마다 고급 모델 1회 해금** 혜택이 제공됩니다. 개인 Gemini API 키를 사이드바에서 등록하면 즉시 `CUSTOM`(개인 키 모드)로 전환되어 모든 고급 모델을 무제한 이용하실 수 있습니다.
