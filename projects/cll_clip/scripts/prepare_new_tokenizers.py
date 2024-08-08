import os
import argparse
import configs
from glob import glob
from clfm.utils import train_new_tokenizer, get_cache_folder
from transformers import AutoTokenizer

NAME_TO_TAG = {
    'openai/clip-vit-base-patch16': 'clip',
    'openai/clip-vit-base-patch32': 'clip',
    'openai/clip-vit-large-patch14': 'clip',
}

parser = argparse.ArgumentParser()
parser.add_argument('--root_format', type=str,
                    default='%s/multilingual_{dataset}/36langs' % configs.corpus_root)
parser.add_argument('--plain_text_format', type=str, 
                    default='{root}/plain_text/*.txt')
parser.add_argument('--save_path_format', type=str, 
                    default='%s/{dataset}/{tag}/{lang}/{vocab_size}' % configs.tokenizer_root)

parser.add_argument('--tokenizer_name', type=str, default='openai/clip-vit-base-patch16', choices=NAME_TO_TAG.keys())
parser.add_argument('--vocab_size', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--oracle', action='store_true')
parser.add_argument('--dataset', type=str, nargs='+', default=['coco'])
args = parser.parse_args()

roots = [args.root_format.format(dataset=dataset) for dataset in args.dataset]
folder_name = '-'.join(args.dataset)

print('root paths:', roots)
print('folder name:', folder_name)
print('tokenizer name:', args.tokenizer_name)

possible_path = os.path.join(get_cache_folder(), args.tokenizer_name.replace('/', '_'))
if os.path.exists(possible_path):
    tokenizer = AutoTokenizer.from_pretrained(possible_path)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

if args.oracle:
    files = [file for root in roots for file in glob(args.plain_text_format.format(root=root))]
    
    save_path = args.save_path_format.format(
        dataset=folder_name,
        tag=NAME_TO_TAG[args.tokenizer_name], 
        lang='oracle',
        vocab_size=args.vocab_size,
    )
    if not os.path.exists(save_path):
        print('- training files:', files)
        train_new_tokenizer(
            tokenizer,
            files=files,
            vocab_size=args.vocab_size,
            save_path=save_path,
            batch_size=args.batch_size,
        )
        print('- Save to', save_path)
else:
    for file in glob(args.plain_text_format.format(root=roots[0])):
        files = [file] + [file.replace(roots[0], roots[_]) for _ in range(1, len(roots))]

        lang = os.path.basename(file).split('.')[0]
        save_path = args.save_path_format.format(
            dataset=folder_name,
            tag=NAME_TO_TAG[args.tokenizer_name], 
            lang=lang,
            vocab_size=args.vocab_size,
        )
        if os.path.exists(save_path):
            continue
        print('- training', files)
        train_new_tokenizer(
            tokenizer,
            files=files,
            vocab_size=args.vocab_size,
            save_path=save_path,
            batch_size=args.batch_size,
        )
        print('- Save to', save_path)

