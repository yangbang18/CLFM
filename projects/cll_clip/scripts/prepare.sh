model=$1
model=${model:-openai/clip-vit-base-patch16}

datasets=(
    coco
    cc3m
)

for dataset in "${datasets[@]}"
do
    python3 scripts/prepare_multimodal_corpus.py --dataset $dataset
    python3 scripts/prepare_plain_text.py --dataset $dataset
    python3 scripts/prepare_new_tokenizers.py --dataset $dataset --vocab_size 10000 --tokenizer_name $model
    python3 scripts/prepare_new_tokenizers.py --dataset $dataset --vocab_size 200000 --oracle --tokenizer_name $model
    python3 scripts/prepare_image_feats.py --dataset $dataset --teacher_model_name $model
    python3 scripts/prepare_text_feats.py --dataset $dataset --teacher_model_name $model
done

python3 scripts/prepare_new_tokenizers.py --dataset coco cc3m --vocab_size 10000 --tokenizer_name $model
python3 scripts/prepare_new_tokenizers.py --dataset coco cc3m --vocab_size 200000 --oracle --tokenizer_name $model
