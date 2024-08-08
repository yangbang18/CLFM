method=$1
arch=$2
trainDatasets=$3
langs=$4
cmd=$5
root=$6

method=${method:-CLL_CLIP}
arch=${arch:-B16}
trainDatasets=${trainDatasets:-coco}
langs=${langs:-}
cmd=${cmd:-}
root=${root:-}

if [ -z ${langs} ];
then
	langs="ar bn cs da de el en es fa fi fil fr he hi hr hu id it ja ko mi nl no pl pt quz ro ru sv sw te th tr uk vi zh"
fi

if [ -z ${root} ];
then
    if [ $method = 'M-CLIP' ];
    then
	root=output/zero_shot
    else
        # root="output/coco_cc3m/MT" if root is empty and trainDatasets="coco,cc3m"
        root=output/${trainDatasets//,/_}/MT
    fi
fi
outputPath=${root}/${method}
echo "output_path: ${outputPath}"

    
if [ ${arch} == "B16" ];
then
    pt=openai/clip-vit-base-patch16
elif [ ${arch} == "B32" ];
then
    pt=openai/clip-vit-base-patch32
elif [ ${arch} == "L14" ];
then
    pt=openai/clip-vit-large-patch14
else
    echo "Error! Only support one of [B32, B16, L14] now, but received ${arch}"
    exit 64
fi

if [ $method = 'M-CLIP' ];
then
    model=M-CLIP/XLM-Roberta-Large-Vit-B-32
    csv_fn_format={dataset}_zeroshot_{task}{postfix}.csv

    # zero-shot evaluation
    bash scripts/eval.sh ${model} coco "${langs}" "" ${outputPath}
    bash scripts/eval.sh ${model} xm3600 "${langs}" "" ${outputPath}
else
    cmd="${cmd} --method ${method} --teacher_model_name ${pt} --base_model_name ${pt} --embedding_name ${pt}"
    csv_fn_format={dataset}_{type}_{task}{postfix}.csv

    echo "cmd: ${cmd}"
    # training
    python3 train.py \
        --use_amp \
        --do_evaluation \
        --val_dataset coco \
        --target_languages $langs \
        --train_datasets ${trainDatasets} \
        --output_path "${outputPath}" \
        ${cmd}
    
    # evaluation
    bash scripts/eval.sh "${outputPath}" coco "${langs}"
    bash scripts/eval.sh "${outputPath}" xm3600 "${langs}"
fi

# gather results
csvPath="./results"
echo "gather results and save them to ${csvPath}"

cmd="--root $(dirname ${root}) --csv_path ${csvPath} --csv_fn_format ${csv_fn_format} --sort"
python3 scripts/gather.py MT --dataset coco --score_fn translated_test_scores.json --eval ${cmd}
python3 scripts/gather.py MT --dataset xm3600 --score_fn test_scores.json ${cmd}
