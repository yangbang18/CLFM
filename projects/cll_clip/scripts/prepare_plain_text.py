import gzip
import os
import argparse
import configs
from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument('--root_format', type=str,
                    default='%s/multilingual_{dataset}/36langs' % configs.corpus_root)
parser.add_argument('--text_corpus_format', type=str, 
                    default='{root}/*.tsv.gz')
parser.add_argument('--save_path_format', type=str, 
                    default='{root}/plain_text/{lang}.txt')

parser.add_argument('--dataset', type=str, default='coco')
args = parser.parse_args()

root = args.root_format.format(dataset=args.dataset)
print('root path:', root)


for file in glob(args.text_corpus_format.format(root=root)):
    fn = os.path.basename(file).split('.')[0].split('_')[-1]
    if '-' in fn:
        source_language, target_language = fn.split('-')
    else:
        source_language = target_language = fn
    
    save_path = args.save_path_format.format(root=root, lang=target_language)
    if os.path.exists(save_path):
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

    print('- saving plain texts to', save_path)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as wf:
        wf.write('\n'.join(targets))
