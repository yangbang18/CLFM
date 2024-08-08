#!/usr/bin/env bash
gpu=$1
gpu=${gpu:-0}

CUDA_VISIBLE_DEVICES=$gpu python3 scripts/analysis_attention.py --arch B16 --method CLL_CLIP --save_fn attention_kl_id0.txt --use_amp
CUDA_VISIBLE_DEVICES=$gpu python3 scripts/analysis_attention.py --arch B16 --method CLL_CLIP_Oracle0_ID1_REG0 --save_fn attention_kl_id1.txt --use_amp
