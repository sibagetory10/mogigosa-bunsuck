#!/usr/bin/env bash
# Oracle Cloud Ubuntu 24.04 (VM.Standard.A1.Flex ARM64 / E2 Micro) 24시간 단일 프로세스 배포 스크립트
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

echo "[1/6] 시스템 패키지(Python venv, Git, 한글 폰트) 설치 중..."
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip python3-venv git fonts-nanum iptables-persistent

echo "[2/6] 가상환경(.venv) 및 Python 패키지 설치 중..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "[3/6] 우분투 방화벽(iptables) 8501 포트 개방..."
if ! sudo iptables -C INPUT -p tcp --dport 8501 -j ACCEPT 2>/dev/null; then
    sudo iptables -I INPUT 1 -p tcp --dport 8501 -j ACCEPT
fi
sudo netfilter-persistent save

echo "[4/6] .streamlit/secrets.toml 확인..."
mkdir -p .streamlit
if [ ! -f ".streamlit/secrets.toml" ]; then
    cat << 'EOF' > .streamlit/secrets.toml
GEMINI_API_KEY = "여기에_GEMINI_API_KEY_입력"
ADMIN_USERNAME = "sibagetory10"
EOF
    echo "⚠️ .streamlit/secrets.toml 템플릿을 생성했습니다. GEMINI_API_KEY를 실제 키로 수정해 주세요!"
fi

echo "[5/6] systemd 서비스(mogigosa.service) 등록 (단일 프로세스 RLock 보장)..."
SERVICE_USER="$(whoami)"
sudo tee /etc/systemd/system/mogigosa.service > /dev/null << EOF
[Unit]
Description=MogiMogi 2022 Curriculum Exam Platform (Streamlit)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
WorkingDirectory=${APP_DIR}
ExecStart=${APP_DIR}/.venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

echo "[6/6] GitHub 자동 업데이트 & 재시작 타이머(mogigosa-autoupdate.timer) 등록..."
chmod +x "${APP_DIR}/scripts/oracle_auto_update.sh"
sudo tee /etc/systemd/system/mogigosa-autoupdate.service > /dev/null << EOF
[Unit]
Description=MogiMogi GitHub Auto-Update & Restart Service
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=${SERVICE_USER}
WorkingDirectory=${APP_DIR}
ExecStart=/usr/bin/env bash ${APP_DIR}/scripts/oracle_auto_update.sh
EOF

sudo tee /etc/systemd/system/mogigosa-autoupdate.timer > /dev/null << EOF
[Unit]
Description=Check GitHub origin/main every 1 minute for MogiMogi auto-update

[Timer]
OnBootSec=1min
OnUnitActiveSec=1min
AccuracySec=10s
Unit=mogigosa-autoupdate.service

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable mogigosa.service
sudo systemctl restart mogigosa.service
sudo systemctl enable mogigosa-autoupdate.timer
sudo systemctl restart mogigosa-autoupdate.timer

echo "✅ 설치 및 자동 업데이트 서비스 등록 완료!"
echo "- 앱 상태 확인: sudo systemctl status mogigosa"
echo "- 자동 업뎃 확인: sudo systemctl status mogigosa-autoupdate.timer"
echo "- 자동 업뎃 로그: journalctl -u mogigosa-autoupdate -f"
echo "- 접속 주소: http://\$(curl -s ifconfig.me):8501"
