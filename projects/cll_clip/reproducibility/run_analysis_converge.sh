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
)

cmd="--save_root reproducibility/analysis_results --save_fn converge_loss.txt --use_amp"
for method in "${methods[@]}"
do
    echo "running $method"
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/analysis_converge.py --arch B16 --method ${method} ${cmd}
done

# we do not use automatic mixed precision in this case to avoid `nan`
cmd="--save_root reproducibility/analysis_results --save_fn converge_fisher.txt --fisher"
for method in "${methods[@]}"
do
    echo "running $method"
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/analysis_converge.py --arch B16 --method ${method} ${cmd}
done
