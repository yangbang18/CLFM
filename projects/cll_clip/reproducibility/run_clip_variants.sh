gpu=$1
gpu=${gpu:-0}

model=openai/clip-vit-base-patch16
outputPath=output/zero_shot/openai_clip_B16
CUDA_VISIBLE_DEVICES=$gpu bash scripts/eval.sh $model xm3600 "" "" $outputPath
CUDA_VISIBLE_DEVICES=$gpu bash scripts/eval.sh $model coco "" "" $outputPath
CUDA_VISIBLE_DEVICES=$gpu bash scripts/eval.sh $model flickr30k "en de fr cs" "" $outputPath

model=M-CLIP/XLM-Roberta-Large-Vit-B-32
outputPath=output/zero_shot/M-CLIP
CUDA_VISIBLE_DEVICES=$gpu bash scripts/eval.sh $model xm3600 "" "" $outputPath
CUDA_VISIBLE_DEVICES=$gpu bash scripts/eval.sh $model coco "" "" $outputPath
CUDA_VISIBLE_DEVICES=$gpu bash scripts/eval.sh $model flickr30k "en de fr cs" "" $outputPath

python3 scripts/gather.py MT --root output --csv_fn_format {dataset}_zeroshot_{task}{postfix}.csv --dataset flickr30k --sort
python3 scripts/gather.py MT --root output --csv_fn_format {dataset}_zeroshot_{task}{postfix}.csv --dataset xm3600 --sort
python3 scripts/gather.py MT --root output --csv_fn_format {dataset}_zeroshot_{task}{postfix}.csv --dataset coco --score_fn translated_test_scores.json --sort
