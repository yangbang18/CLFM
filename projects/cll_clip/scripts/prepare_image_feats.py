import numpy as np
import gzip
import os
import sys
import argparse
import json
import configs
from clfm import Framework
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, vision_root, image_relative_paths) -> None:
        super().__init__()
        self.vision_root = vision_root
        self.image_relative_paths = image_relative_paths
    
    def __getitem__(self, index):
        image_path = os.path.join(self.vision_root, self.image_relative_paths[index])
        return Image.open(image_path).convert("RGB")

    def __len__(self):
        return len(self.image_relative_paths)

    def collate_fn(self, batch):
        return batch


parser = argparse.ArgumentParser()
parser.add_argument('--read_format', type=str, 
                    default='%s/multilingual_{dataset}/36langs/triplet/{dataset}_vision-en-en.tsv.gz' % configs.corpus_root)
parser.add_argument('--save_format', type=str, 
                    default='%s/multilingual_{dataset}/36langs/embeddings/{teacher_model_name}_vision_embeddings.npy' % configs.corpus_root)

parser.add_argument('--json_format', type=str, 
                    default='%s/{dataset}/en/train.json' % configs.annotation_root)
parser.add_argument('--text_corpus_format', type=str, 
                    default='%s/multilingual_{dataset}/36langs/*.tsv.gz' % configs.corpus_root)

parser.add_argument('--teacher_model_name', type=str, default='openai/clip-vit-base-patch16')
parser.add_argument('--dataset', type=str, default='coco')
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

save_path = args.save_format.format(
    dataset=args.dataset,
    teacher_model_name=args.teacher_model_name.replace("/", "_")
)
os.makedirs(os.path.dirname(save_path), exist_ok=True)
if os.path.exists(save_path):
    print(save_path, 'exists!!!')
    sys.exit(0)
print('save path:', save_path)

json_path = args.json_format.format(dataset=args.dataset)
json_data = json.load(open(json_path))

read_path = args.read_format.format(dataset=args.dataset)
if not os.path.exists(read_path):
    raise FileNotFoundError(f"please run `python scripts/prepare_multimodal_corpus.py --dataset {args.dataset}` first")

print('read from', read_path)
assert os.path.exists(read_path)
rpaths_set = set()
rpaths_list = []
with gzip.open(read_path, 'rt', encoding='utf8') as f:
    count = 0
    index = 0
    for line in f:
        text = line.strip()
        rpath, caption, *_ = text.split('\t')

        orig_caption = json_data[count]['caption'].strip().replace('\n', ' ')
        assert caption == orig_caption, f"{caption} {orig_caption}"
        if rpath not in rpaths_set:
            rpaths_set.add(rpath)
            rpaths_list.append(rpath)
            index += 1
        count += 1

image_dataset = ImageDataset(
    vision_root=configs.image_video_root[args.dataset],
    image_relative_paths=rpaths_list,
)
loader = DataLoader(image_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=image_dataset.collate_fn)
model = Framework(args.teacher_model_name)

all_embeddings = []
for images in tqdm(loader):    
    embeddings = model.encode(images, batch_size=len(images), show_progress_bar=False, convert_to_numpy=True)
    all_embeddings.append(embeddings)

all_embeddings = np.concatenate(all_embeddings, axis=0)
np.save(save_path, all_embeddings)
