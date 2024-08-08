gpu=$1
gpu=${gpu:-0}

model=openai/clip-vit-base-patch16
outputPath=output/zero_shot/openai_clip_B16
CUDA_VISIBLE_DEVICES=$gpu bash scripts/eval.sh $model flickr30k "en de fr cs" "" $outputPath

model=M-CLIP/XLM-Roberta-Large-Vit-B-32
outputPath=output/zero_shot/M-CLIP
CUDA_VISIBLE_DEVICES=$gpu bash scripts/eval.sh $model flickr30k "en de fr cs" "" $outputPath

python3 scripts/gather.py MT --root output --csv_fn_format {dataset}_zeroshot_{task}{postfix}.csv --dataset flickr30k --sort


methods=(
CLL_CLIP
CLL_CLIP_with_TEIR
MLA
MLA_with_TEIR
)

for method in "${methods[@]}"
do
    # `hu` is the last language to be learned
    model=output/coco/CL/B16_$method/order222/hu
    CUDA_VISIBLE_DEVICES=$gpu bash scripts/eval.sh $model flickr30k "en de fr cs"
done
python3 scripts/gather.py CL --dataset flickr30k --skip_cl_csv --sort

