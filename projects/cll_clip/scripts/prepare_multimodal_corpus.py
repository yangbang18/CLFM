import gzip
import os
import argparse
import json
import configs
from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument('--root_format', type=str,
                    default='%s/multilingual_{dataset}/36langs' % configs.corpus_root)
parser.add_argument('--text_corpus_format', type=str, 
                    default='{root}/*.tsv.gz')
parser.add_argument('--vision_text_corpus_format', type=str, 
                    default='{root}/vision_text/{dataset}_vision-{target}.tsv.gz')
parser.add_argument('--triplet_corpus_format', type=str, 
                    default='{root}/triplet/{dataset}_vision-{source}-{target}.tsv.gz')
parser.add_argument('--json_format', type=str, 
                    default='%s/{dataset}/en/train.json' % configs.annotation_root)

parser.add_argument('--dataset', type=str, default='coco')
args = parser.parse_args()

root = args.root_format.format(dataset=args.dataset)
print('root path:', root)

json_path = args.json_format.format(dataset=args.dataset)
print('load json data from', json_path)
json_data = json.load(open(json_path))


if args.dataset == 'cc3m':
    line2idx = json.load(open(os.path.join(root, 'line2idx.json')))
    existed_lines = [item['image_id'] for item in json_data]
    existed_ids = [line2idx[str(line)] for line in existed_lines]


for file in glob(args.text_corpus_format.format(root=root)):
    fn = os.path.basename(file).split('.')[0].split('_')[-1]
    if '-' in fn:
        source_language, target_language = fn.split('-')
    else:
        source_language = target_language = fn

    # check if the files have been saved
    this_vision_text_corpus_path = args.vision_text_corpus_format.format(
        root=root, dataset=args.dataset, target=target_language)
    this_triplet_corpus_path = args.triplet_corpus_format.format(
        root=root, dataset=args.dataset, source=source_language, target=target_language)
    
    if os.path.exists(this_vision_text_corpus_path) and os.path.exists(this_triplet_corpus_path):
        print(f'- {target_language} have been processed')
        continue
    
    print('- reading', file)
    sources = []
    targets = []
    with gzip.open(file, 'rt', encoding='utf8') as f:
        for line in f:
            text = line.strip()
            sources.append(text.split('\t')[0])
            targets.append(text.split('\t')[-1])
    
    if args.dataset == 'cc3m':
        sources = [sources[_id] for _id in existed_ids]
        targets = [targets[_id] for _id in existed_ids]

    assert len(targets) == len(json_data), \
        f"json data has {len(json_data)} lines, but the text file has {len(targets)} lines"

    # ========== vision & target text ============
    os.makedirs(os.path.dirname(this_vision_text_corpus_path), exist_ok=True)

    print('- saving to', this_vision_text_corpus_path)
    with gzip.open(this_vision_text_corpus_path, 'wt', encoding='utf8') as f:
        for line, target in zip(json_data, targets):
            line = f"{line['image']}\t{target}\n"
            f.write(line)
        
        print(line)
    
    # ========== vision & source text & target_text ============
    os.makedirs(os.path.dirname(this_triplet_corpus_path), exist_ok=True)

    print('- saving to', this_triplet_corpus_path)
    with gzip.open(this_triplet_corpus_path, 'wt', encoding='utf8') as f:
        for line, source, target in zip(json_data, sources, targets):
            line = f"{line['image']}\t{source}\t{target}\n"
            f.write(line)
        
        print(line)

if args.dataset == 'cc3m':
    for format in [args.vision_text_corpus_format, args.triplet_corpus_format]:
        path = os.path.dirname(format).format(root=root)
        with open(os.path.join(path, 'existed_ids.json'), 'w') as wf:
            json.dump(existed_ids, wf)
