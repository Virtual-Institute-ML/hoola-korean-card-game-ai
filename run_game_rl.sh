#!/usr/bin/env bash
python3 run_game.py \
    --p0 rl \
    --p1 human \
    --rl-checkpoint checkpoints/best.pt \
    --training-record records_human/human_vs_rl_v1.npz
