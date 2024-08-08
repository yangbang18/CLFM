model=$1
outputPath=$2
other=$3
outputPaht=${outputPath:-}
other=${other:-}

python infer_caption.py --auto --model ${model} --output_path "${outputPath}" --dataset coco --lang en ${other}
#python infer_caption.py --auto --model ${model} --output_path "${outputPath}" --dataset flickr30k --lang en ${other}
python infer_caption.py --auto --model ${model} --output_path "${outputPath}" --dataset flickr30k --lang zh ${other}
python infer_caption.py --auto --model ${model} --output_path "${outputPath}" --dataset flickr30k --lang de ${other}
python infer_caption.py --auto --model ${model} --output_path "${outputPath}" --dataset flickr30k --lang fr ${other}
python infer_caption.py --auto --model ${model} --output_path "${outputPath}" --dataset msrvtt --lang en ${other}
python infer_caption.py --auto --model ${model} --output_path "${outputPath}" --dataset vatex --lang zh ${other}
#python infer_caption.py --auto --model ${model} --output_path "${outputPath}" --dataset vatex --lang en ${other}

# bash scripts/caption.sh zeronlg-4langs-vc output/zeronlg-4langs-vc
