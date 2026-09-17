# Hoola AI Web MVP

이 폴더는 기존 `hoola_ai_v1.1` Python 프로젝트 **안에 `web/` 폴더로 복사**해서 쓰는 것을 전제로 합니다.
기존 `hoola/`, `agents/`, `rl/` 코드는 수정하지 않습니다.

## 1. 폴더 배치

프로젝트가 다음처럼 보이면 됩니다.

```text
hoola_ai_v1.1/
├── hoola/
├── agents/
├── rl/
├── run_game.py
├── ...
└── web/
    ├── backend/
    └── frontend/
```

이 배포본의 `backend/`, `frontend/`를 통째로 `web/` 아래에 넣으세요.

## 2. Backend

프로젝트 root에서:

```bash
python3 -m pip install fastapi 'uvicorn[standard]' pydantic
uvicorn web.backend.app:app --reload --host 127.0.0.1 --port 8000
```

Heuristic 상대는 checkpoint 없이 바로 동작합니다.

RL 상대를 쓰려면 먼저 환경변수를 지정합니다.

Linux/macOS:

```bash
export HOOLA_RL_CHECKPOINT="checkpoints/rl_v1_small/.../best.pt"
uvicorn web.backend.app:app --reload --host 127.0.0.1 --port 8000
```

Windows PowerShell:

```powershell
$env:HOOLA_RL_CHECKPOINT="checkpoints/rl_v1_small/.../best.pt"
uvicorn web.backend.app:app --reload --host 127.0.0.1 --port 8000
```

API 확인:

```text
http://127.0.0.1:8000/docs
```

## 3. React frontend

새 terminal에서:

```bash
cd web/frontend
npm install
npm run dev
```

브라우저:

```text
http://localhost:5173
```

Vite가 `/api` 요청을 자동으로 FastAPI `:8000`으로 proxy합니다.

## 학습용 gameplay record 저장

이 버전은 웹에서 **끝까지 완료된 게임**을 기존 `TrainingRecorder`와 동일한
`hoola-training-v1` 형식으로 자동 저장합니다. Human과 AI 양쪽 decision을 모두
기록하지만, 기존 `HumanDataset` loader는 `agent_types == "human"`인 행동만 자동으로
골라 학습에 사용합니다. 따라서 별도 변환이 필요 없습니다.

기본 저장 위치(프로젝트 root에서 backend를 실행한 경우):

```text
records/web/
├── web_human_vs_rl_YYYYMMDD_HHMMSS_...npz
├── web_human_vs_rl_YYYYMMDD_HHMMSS_...json
├── web_human_vs_heuristic_YYYYMMDD_HHMMSS_...npz
└── web_human_vs_heuristic_YYYYMMDD_HHMMSS_...json
```

게임을 중간에 메뉴로 나가거나 서버를 종료한 경우에는 불완전한 outcome을 학습에
섞지 않도록 파일을 저장하지 않습니다. 게임이 terminal 상태가 되었을 때만 한 번 저장합니다.

저장 위치를 바꾸려면 backend 실행 전에:

```bash
export HOOLA_WEB_RECORD_DIR="records/web"
```

기록 기능을 잠시 끄려면:

```bash
export HOOLA_WEB_RECORDING=0
```

이후 기존 학습에서 그대로 사용할 수 있습니다. 예:

```bash
python3 train_rl.py \
    ... \
    --human-data records/web/
```

기존 CLI human records와 웹 records를 동시에 쓰고 싶으면 같은 상위 폴더 아래에 두거나,
학습용 폴더에 함께 모아 `--human-data`로 지정하면 됩니다.

## 현재 MVP UX

- Human은 Player 0으로 시작
- Heuristic 또는 RL 선택
- stock pile 클릭 = draw
- DISCARD phase에서는 손의 카드를 직접 클릭해 버림
- MELD / LAYOFF / THANK YOU / PASS는 중앙 action button
- AI의 패는 항상 card back으로 숨김
- public meld / discard / stock / registration 표시
- terminal win/loss/draw 표시
- 최근 action log 표시

## 다음 단계 권장 순서

1. MVP 실제 플레이로 규칙/UI 버그 제거
2. 멜드 선택을 action-button 대신 카드 multi-select UX로 개선
3. 웹 gameplay training record 검증 및 데이터 축적
4. PWA manifest + service worker
5. `best.pt -> ONNX` 변환 후 browser/offline inference
6. Capacitor Android/iOS 패키징
