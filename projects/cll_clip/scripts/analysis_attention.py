'''This file is responsible for comparing
    the attention distributions of the final model and the intermediate models.
    
    We report the following Kullback-Leibler (KL) Divergence:
        KL(final || intermediate)

    Lower values indicate that 
    the model's attention pattern is more stable and consistent during continual learning.
        
    See reproducibility/run_analysis_attention.sh for a running example
'''
import gzip
import argparse
import os
import configs
import torch
from tqdm import tqdm
from clfm import Framework
from clfm.utils import batch_to_device

parser = argparse.ArgumentParser()
parser.add_argument('--model_root', type=str, default='output/coco/CL')
parser.add_argument('--arch', type=str, default='B16', choices=['B16', 'B32', 'L14'])
parser.add_argument('--method', type=str, default='CLL_CLIP')
parser.add_argument('--order', type=str, default='order222')
parser.add_argument('--data_root', type=str, default='data/corpus/multilingual_coco/36langs/triplet')
parser.add_argument('--data_fn_format', type=str, default='coco_vision-en-{lang}.tsv.gz')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_samples', type=int, default=-1)
parser.add_argument('--save_root', type=str, default='reproducibility/analysis_results')
parser.add_argument('--save_fn', type=str, default='attention_kl_id0.txt')
parser.add_argument('--use_amp', action='store_true')
args = parser.parse_args()


def load(lang, n_samples):
    fn = args.data_fn_format.format(lang=lang)
    path = os.path.join(args.data_root, fn)
    print('load data from', path)
    captions = []
    count = 0
    with gzip.open(path, 'rt', encoding='utf8') as f:
        for line in f:
            vision, _, caption = line.strip().split('\t')
            captions.append(caption)
            count += 1
            if n_samples != -1 and count == n_samples:
                break
        
    return captions


@torch.no_grad()
def run(model, texts, batch_size=64, tokenize_model=None):
    if tokenize_model is None:
        tokenize_model = model
    
    model.eval()
    model = model.to(model.device)
    
    n_batches = len(texts) // batch_size
    if n_batches * batch_size != len(texts):
        n_batches += 1

    raw_attentions = []
    for n in tqdm(range(n_batches)):
        start = n * batch_size
        end = (n + 1) * batch_size

        tokenized = tokenize_model.tokenize(texts[start:end])
        features = batch_to_device(tokenized, model.device)
        if args.use_amp:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model(features)
        else:
            outputs = model(features)
        
        # (batch_size, L, H, len, len)
        attentions = torch.stack([item.cpu() for item in outputs['attentions']], dim=1)
        attentions = attentions.mean((1, 2))
        assert attentions.ndim == 3

        for pos_eos, attns in zip(features['eos_ids'], attentions):
            raw_attentions.append(attns[pos_eos, :pos_eos+1].view(-1))
        
    return raw_attentions


tasks = getattr(configs, f'xm3600_{args.order}', None)
assert tasks is not None, f'xm3600_{args.order} can not be found in configs'
final_task_id = len(tasks) - 1
final_task = tasks[final_task_id]
final_model_path = os.path.join(args.model_root, f"{args.arch}_{args.method}", args.order, final_task)
final_model = Framework(final_model_path)
print('final task:', final_task)

os.makedirs(args.save_root, exist_ok=True)

f = open(os.path.join(args.save_root, args.save_fn), 'w')
header = '\t'.join(['task_id', 'kl_div'])
f.write(f'{header}\n')

for task_id in tqdm(range(final_task_id)):
    this_task = tasks[task_id]
    this_model_path = os.path.join(args.model_root, f"{args.arch}_{args.method}", args.order, this_task)
    model = Framework(this_model_path)
    print('model_path:', this_model_path)
    print(f'the model has learned {task_id+1} tasks')
    print('n_samples:', args.n_samples)

    data = load(this_task, args.n_samples)

    raw_attentions_intermediate = run(
        model=model, 
        texts=data,
        batch_size=args.batch_size,
    )

    raw_attentions_final = run(
        model=final_model, 
        tokenize_model=model,
        texts=data, 
        batch_size=args.batch_size,
    )

    all_kl_div = []
    for raw_attn1, raw_attn2 in tqdm(zip(raw_attentions_intermediate, raw_attentions_final)):
        kl_div = (raw_attn2 * torch.log(raw_attn2 / raw_attn1))
        all_kl_div.append(kl_div)
    avg_kl_div = torch.cat(all_kl_div).mean().item()
    
    save_content = f'{task_id}\t{avg_kl_div}'
    print(save_content)
    f.write(f'{save_content}\n')
    torch.cuda.empty_cache()

f.close()
