cd /home/jhha/card_game/hoola_ai_v1.1

export HOOLA_RL_CHECKPOINT="checkpoints/rl_v1_small/rl_v1_260913_t1/best.pt"
export HOOLA_WEB_RECORD_DIR="records_human"

python3 -m uvicorn web.backend.app:app \
    --reload \
    --host 127.0.0.1 \
    --port 8000
