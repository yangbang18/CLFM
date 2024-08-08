#!/usr/bin/env bash
gpu=$1
gpu=${gpu:-0}

if [[ ! -d output/coco_cc3m/CL/B16_CLL_CLIP_with_TEIR ]];
then
    CUDA_VISIBLE_DEVICES=$gpu bash scripts/run_cl.sh CLL_CLIP_with_TEIR B16 coco,cc3m
fi

lang="he"

CUDA_VISIBLE_DEVICES=$gpu python3 infer_retrieval.py \
    --model openai/clip-vit-base-patch16 \
    --clip_model_name openai/clip-vit-base-patch16 \
    --output_path output/translate_test/zero_shot/openai_clip_B16 \
    --mid_path translated_to_en \
    --lang $lang \
    --dataset xm3600

python3 scripts/gather.py MT \
    --root output/translate_test \
    --csv_path reproducibility/analysis_results \
    --csv_fn_format translated.csv \
    --score_fn translated_test_scores.json \
    --retrieval_metrics img_r1 txt_r1

trainDatasets=(
    coco
    coco_cc3m
)
for trainDataset in "${trainDatasets[@]}"
do
    echo "running ${trainDataset}"
    CUDA_VISIBLE_DEVICES=$gpu python3 infer_retrieval.py \
        --model output/${trainDataset}/CL/B16_CLL_CLIP_with_TEIR/order222/$lang \
        --output_path output/translate_test/${trainDataset}/B16_CLL_CLIP_with_TEIR \
        --mid_path translated_to_en \
        --lang $lang \
        --dataset xm3600 \
        --do_fusion \
        --fusion_types score
done

python3 scripts/gather.py MT \
    --root output/translate_test \
    --csv_path reproducibility/analysis_results \
    --csv_fn_format fused.csv \
    --score_fn fused_test_scores.json \
    --retrieval_metrics img_r1 txt_r1
