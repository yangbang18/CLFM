model=$1
dataset=$2
langs=$3
cmd=$4
outputPath=$5

dataset=${dataset:-xm3600}
langs=${langs:-}
cmd=${cmd:-}
outputPath=${outputPath:-}

if [[ ${dataset} == coco && ${langs} != "en" ]];
then
	cmd="${cmd} --mid_path translated"
fi

if [[ ${model} =~ "openai" ]];
then
	cmd="${cmd} --clip_model_name ${model}"
fi

if [[ -z "${langs}" ]];
then
	langs="ar bn cs da de el en es fa fi fil fr he hi hr hu id it ja ko mi nl no pl pt quz ro ru sv sw te th tr uk vi zh"
fi

echo cmd: ${cmd}
echo langs: ${langs}

python3 infer_retrieval.py \
	--model ${model} \
	--dataset ${dataset} \
	--output_path "${outputPath}" \
	--langs ${langs} \
	--skip_if_exists \
	--wisely_activate_adapter \
	--wisely_activate_adapter_fusion \
    ${cmd}
