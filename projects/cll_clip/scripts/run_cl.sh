method=$1
arch=$2
trainDatasets=$3
order=$4
cmd=$5
root=$6

method=${method:-CLL_CLIP}
arch=${arch:-B16}
trainDatasets=${trainDatasets:-coco}
order=${order:-order222}
cmd=${cmd:-}
root=${root:-}

cmd="${cmd} --method ${method}"
if [[ ${arch} == "B16" ]];
then
    pt=openai/clip-vit-base-patch16
elif [[ ${arch} == "B32" ]];
then
    pt=openai/clip-vit-base-patch32
elif [[ ${arch} == "L14" ]];
then
    pt=openai/clip-vit-large-patch14
else
    echo "Error! Only support one of [B32, B16, L14] now, but received ${arch}"
    exit 64
fi
cmd="${cmd} --teacher_model_name ${pt} --base_model_name ${pt} --embedding_name ${pt}"

if [[ -z ${root} ]];
then
    # root="output/coco_cc3m/CL" if root is empty and trainDatasets="coco,cc3m"
    root=output/${trainDatasets//,/_}/CL
fi
outputPath=${root}/${arch}_${method}/${order}

############
echo "cmd: ${cmd}"
echo "output_path: ${outputPath}"
############

if [ "$order" == "order222" ];
then
    langs=(en it de fil tr uk nl bn he fi sv quz fa mi fr el zh id sw no hi da te th pt ru ro pl hr es vi ar cs ko ja hu)
else
    echo "Error! Only support order222 now"
    exit 64
fi

len=`echo ${#langs[@]}`

for ((i=0;i<$len;i++))
do
    lang=`echo ${langs[$i]}`
    echo "$i $lang"

    thisPath=${outputPath}/"${lang}"

    if [ $i -gt 0 ];
    then
        # load pre-trained weights from the last task
        pretrained=${outputPath}/`echo ${langs[$[i-1]]}`
    else
        # run the first task
        pretrained=""
    fi

    # training
    python3 train.py \
    --use_amp \
    --do_evaluation \
    --val_dataset coco \
    --target_languages ${lang} \
    --train_datasets ${trainDatasets//,/ } \
    --student_model_name "${pretrained}" \
    --output_path "${thisPath}" \
    ${cmd}

    # we only evaluate the model on the current and previous languages to save time
    prevLangs=${langs[@]:0:$[i+1]}
    bash scripts/eval.sh "${thisPath}" coco "${prevLangs}"
    bash scripts/eval.sh "${thisPath}" xm3600 "${prevLangs}"

    # # evaluate the model on all languages (useful when you want to measure `forward transfer`)
    # bash scripts/eval.sh "${thisPath}" coco
    # bash scripts/eval.sh "${thisPath}" xm3600
done


# gather results
csvPath="./results"
echo "gather results and save them to ${csvPath}"

cmd="--root $(dirname ${root}) --csv_path ${csvPath} --CL_order ${order} --record_Fen --sort"
python3 scripts/gather.py CL --dataset coco --score_fn translated_test_scores.json --eval ${cmd}
python3 scripts/gather.py CL --dataset xm3600 --score_fn test_scores.json ${cmd}

