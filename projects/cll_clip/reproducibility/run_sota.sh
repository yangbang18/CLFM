#!/usr/bin/env bash
gpu=$1
gpu=${gpu:-0}

# Run Joint-Training Models
methods=(
    M-CLIP
    JointTrain_CLL_CLIP
)
for method in "${methods[@]}"
do
    echo "running $method"
    CUDA_VISIBLE_DEVICES=$gpu bash scripts/run_mt.sh $method
done

# Run SOTA Continual Learning Models
methods=(
    CLL_CLIP
    CLL_CLIP_with_TEIR
    oEWC
    oEWC_with_TEIR
    ER
    ER_with_TEIR
    DER
    DER_with_TEIR
    MLA
    MLA_with_TEIR
    P_Tuning
    P_Tuning_with_TEIR
    LoRA
    LoRA_with_TEIR
    DualPrompt
    DualPrompt_with_TEIR
    CodaPrompt
    CodaPrompt_with_TEIR
)

for method in "${methods[@]}"
do
    echo "running $method"
    CUDA_VISIBLE_DEVICES=$gpu bash scripts/run_cl.sh $method
done
