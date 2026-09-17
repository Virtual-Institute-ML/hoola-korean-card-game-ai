#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash generate_heuristic_data.sh          # 1000 games
#   bash generate_heuristic_data.sh 2000     # 2000 games
#
# Heuristic v1 demonstration generator.
# Seats alternate every game to reduce P0/P1 bias.

N_GAMES="${1:-1000}"
OUT_DIR="records_heuristic"
BASE_RECORD="${OUT_DIR}/heuristic_demo.npz"
LOG_FILE="${OUT_DIR}/generation.log"

mkdir -p "${OUT_DIR}"

echo "========================================================================"
echo "Generating ${N_GAMES} Heuristic demonstration games"
echo "Output : ${OUT_DIR}/heuristic_demo_XXX.npz (+ .json)"
echo "Log    : ${LOG_FILE}"
echo "========================================================================"

START_TIME=$(date +%s)

for ((i=1; i<=N_GAMES; i++)); do
    # Alternate seats to reduce first/second-player bias.
    if (( i % 2 == 1 )); then
        P0="heuristic"
        P1="heuristic"
    else
        P0="heuristic"
        P1="heuristic"
    fi

    python3 run_game.py \
        --p0 "${P0}" \
        --p1 "${P1}" \
        --quiet \
        --training-record "${BASE_RECORD}" \
        >> "${LOG_FILE}" 2>&1

    if (( i % 50 == 0 || i == N_GAMES )); then
        NOW=$(date +%s)
        ELAPSED=$((NOW - START_TIME))
        echo "[${i}/${N_GAMES}] completed | elapsed ${ELAPSED}s"
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "========================================================================"
echo "Done: ${N_GAMES} games in ${ELAPSED}s"
echo "Records saved under ${OUT_DIR}/"
echo "========================================================================"
