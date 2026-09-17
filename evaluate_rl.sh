#!/usr/bin/env bash
python3 evaluate_rl.py \
    checkpoints/best.pt \
    --games 1000 \
    --opponent heuristic
