method=$1
arch=$2
trainDatasets=$3
valDataset=$4
cmd=$5
root=$6

method=${method:-CLL_CLIP}
arch=${arch:-B16}
trainDatasets=${trainDatasets:-coco}
valDataset=${valDataset:-coco}
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

if [ ${valDataset} = 'flickr30k' ];
then
	cmd="${cmd} --do_evaluation --val_dataset flickr30k --val_file_format data/annotations/{dataset}/{lang}/val.json"
	langs=(en de fr cs)
else
	cmd="${cmd} --do_evaluation --val_dataset coco"
	langs=(ar bn cs da de el en es fa fi fil fr he hi hr hu id it ja ko mi nl no pl pt quz ro ru sv sw te th tr uk vi zh)
fi

if [ -z ${root} ]
then
    # root="output/coco_cc3m/ST" if root is empty and trainDatasets="coco,cc3m"
    root=output/${trainDatasets//,/_}/ST
fi

############
echo "cmd: ${cmd}"
############

for lang in "${langs[@]}"
do
	outputPath=${root}/${method}/${lang}
    python3 train.py \
		--use_amp \
		--target_languages ${lang} \
		--train_datasets ${trainDatasets} \
		--output_path "${outputPath}" \
		${cmd}
	
	# evaluation
	bash scripts/eval.sh "${outputPath}" coco "${lang}"
	bash scripts/eval.sh "${outputPath}" xm3600 "${lang}"
done

# gather results
time=$(date "+%Y%m%d-%H%M%S")
csvPath="./results/${time}"
echo "gather results and save them to ${csvPath}"

cmd="--root $(dirname ${root}) --csv_path ${csvPath}"
python3 scripts/gather.py ST --dataset coco --score_fn translated_test_scores.json --eval ${cmd}
python3 scripts/gather.py ST --dataset xm3600 --score_fn test_scores.json ${cmd}
