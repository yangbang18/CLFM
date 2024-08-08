import numpy as np
import gzip
import os
import sys
import argparse
import configs
from clfm import Framework

parser = argparse.ArgumentParser()
parser.add_argument('--read_format', type=str, 
                    default='%s/multilingual_{dataset}/36langs/{dataset}_en.tsv.gz' % configs.corpus_root)
parser.add_argument('--save_format', type=str, 
                    default='%s/multilingual_{dataset}/36langs/embeddings/{teacher_model_name}_sentence_embeddings.npy' % configs.corpus_root)

parser.add_argument('--teacher_model_name', type=str, default='openai/clip-vit-base-patch16')
parser.add_argument('--dataset', type=str, default='coco')
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

save_path = args.save_format.format(
    dataset=args.dataset, 
    teacher_model_name=args.teacher_model_name.replace('/', '_')
)
if os.path.exists(save_path):
    print(save_path, 'exists!!!')
    sys.exit(0)
print('save path:', save_path)

os.makedirs(os.path.dirname(save_path), exist_ok=True)

read_path = args.read_format.format(dataset=args.dataset)
data = []
with gzip.open(read_path, 'rt', encoding='utf8') as f:
    for line in f:
        text = line.strip()
        data.append(text)

model = Framework(args.teacher_model_name)
sentence_embeddings = model.encode(data, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True)
np.save(save_path, sentence_embeddings)
