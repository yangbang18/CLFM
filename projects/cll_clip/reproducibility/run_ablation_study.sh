#!/usr/bin/env bash
gpu=$1
gpu=${gpu:-0}

methods=(
    CLL_CLIP
    CLL_CLIP_Oracle0_ID1_REG0
    CLL_CLIP_Oracle0_ID0_REG1_g
    CLL_CLIP_Oracle0_ID0_REG1_d
    CLL_CLIP_Oracle0_ID0_REG1_gd
    CLL_CLIP_with_TEIR
    CLL_CLIP_Oracle1_ID1_REG1_gd
    Ablation_trainAll
    Ablation_onlyCrossModal
    Ablation_onlyCrossLingual
)

for method in "${methods[@]}"
do
    echo "running $method"
    CUDA_VISIBLE_DEVICES=$gpu bash scripts/run_cl.sh $method
done
