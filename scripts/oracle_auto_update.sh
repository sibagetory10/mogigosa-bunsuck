#!/usr/bin/env bash
# GitHub origin/main 변경 감지 시 자동으로 fast-forward pull 및 mogigosa.service 재시작
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

LOCK_FILE="/tmp/mogigosa_autoupdate.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    exit 0
fi

git fetch origin main --quiet

LOCAL_HASH="$(git rev-parse HEAD)"
REMOTE_HASH="$(git rev-parse origin/main)"

if [ "$LOCAL_HASH" = "$REMOTE_HASH" ]; then
    exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 새 업데이트 감지: ${LOCAL_HASH:0:7} -> ${REMOTE_HASH:0:7}"
CHANGED_FILES="$(git diff --name-only "$LOCAL_HASH" "$REMOTE_HASH")"

# data/curriculum_2022.json 로컬 변경이 남아 있으면 exam_db/ncic_standards.json으로 백업 후 원복해 pull 충돌 방지
if ! git diff --quiet -- data/curriculum_2022.json; then
    echo "🔄 data/curriculum_2022.json 로컬 성취기준을 exam_db/ncic_standards.json으로 자동 이관 중..."
    .venv/bin/python -c "
import json, pathlib
cur = json.loads(pathlib.Path('data/curriculum_2022.json').read_text(encoding='utf-8'))
rt_path = pathlib.Path('exam_db/ncic_standards.json')
rt_path.parent.mkdir(parents=True, exist_ok=True)
rt = json.loads(rt_path.read_text(encoding='utf-8')) if rt_path.exists() else {'version': '2022-33', 'categories': {}}
rt_cats = rt.setdefault('categories', {})
for k, v in (cur.get('categories') or {}).items():
    stds = v.get('standards') or []
    if stds:
        rt_cats[k] = stds
rt_path.write_text(json.dumps(rt, ensure_ascii=False, indent=2), encoding='utf-8')
" || true
    git checkout -- data/curriculum_2022.json
fi

if ! git pull --ff-only origin main; then
    echo "❌ git pull --ff-only 실패 (로컬 변경/충돌 확인 필요)"
    exit 1
fi

if echo "$CHANGED_FILES" | grep -qx "requirements.txt"; then
    echo "📦 requirements.txt 변경 감지 → 패키지 업데이트 중..."
    .venv/bin/pip install -r requirements.txt
fi

echo "🔄 mogigosa.service 재시작 중..."
sudo systemctl restart mogigosa.service
echo "✅ 자동 업데이트 및 재시작 완료 (${REMOTE_HASH:0:7})"
