'''Running Examples:
python3 reproducibility/plot_token_embedding_distribution.py --plot_pretrained
python3 reproducibility/plot_token_embedding_distribution.py --method CLL_CLIP_with_TEIR
'''
import os
import argparse
import configs
import torch
import numpy as np
import matplotlib.pyplot as plt
from clfm import Constants
from clfm.models import Transformer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_pretrained', action='store_true')
    parser.add_argument('--arch', type=str, default='B16', choices=['B16', 'B32', 'L14'])
    parser.add_argument('--method', type=str, default='CLL_CLIP_with_TEIR')
    parser.add_argument('--root', type=str, default='output/coco/CL')
    parser.add_argument('--order', type=str, default='order222')
    parser.add_argument('--save_path', type=str, default=os.path.join(os.path.dirname(__file__), 'plot_results'))
    parser.add_argument('--save_fn', type=str, help='If not specified, default to appendix_ted_{method}.png')
    parser.add_argument('--figsize', type=float, nargs='+', default=[12, 16])
    parser.add_argument('--bins', type=int, default=1000)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    str_mu = r'$\mu$'
    str_sigma = r'$\sigma$'

    if args.plot_pretrained:
        fig = plt.figure(figsize=(5, 6))
        names = [
            'openai/clip-vit-base-patch32',
            'openai/clip-vit-base-patch16',
            'openai/clip-vit-large-patch14',
            'bert-base-uncased', 
            't5-base',
            'facebook/bart-large',
        ]
        xlims = [
            (-0.08, 0.08),
            (-0.08, 0.08),
            (-0.08, 0.08),
            (-0.2, 0.2),
            (-150, 150),
            (-0.6, 0.6),
        ]
        for i, (name, xlim) in enumerate(zip(names, xlims)):
            model = Transformer(name)
            try:
                embeddings_weights = model.auto_model.get_input_embeddings().weight
            except:
                embeddings_weights = model.auto_model.text_model.embeddings.token_embedding.weight
            mu = embeddings_weights.mean().item()
            sigma = embeddings_weights.std().item()
            print(name, mu, sigma)

            ax = plt.subplot(3, 2, i+1)
            ax.set_xlim(xlim)
            ax.set_yticks([])
            ax.set_xlabel(f'{name}\n{str_mu}={mu:.3f}, {str_sigma}={sigma:.3f}')
            
            count, bins, _ = ax.hist(embeddings_weights.flatten().detach().numpy(), bins=args.bins, density=True)
            ax.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
        
        plt.subplots_adjust(left=0.04, right=0.96, bottom=0.1, top=0.99, hspace=0.6, wspace=0.2)
        if args.save_fn is None:
            args.save_fn = f'appendix_ted_pretrained.png'
    else:
        langs = getattr(configs, f'xm3600_{args.order}', None)
        if langs is None:
            print(f'The order {args.order} is not found in configs')
            exit(0)
        
        assert len(langs) == 36, f'there should be 36 languages, as in the XM3600 dataset'

        fig = plt.figure(figsize=args.figsize)

        for i, lang in enumerate(langs):
            path = os.path.join(args.root, f"{args.arch}_{args.method}", args.order, lang, '0_AdapterEncoder', Constants.EMB_CKPT_FN)
            assert os.path.exists(path), f'Path {path} does not exist'
            embeddings_weights = torch.load(path, map_location='cpu')['word_embeddings.weight']
            mu = embeddings_weights.mean().item()
            sigma = embeddings_weights.std().item()
            print(lang, mu, sigma)

            ax = plt.subplot(9, 4, i+1)
            ax.set_xlim(-0.08, 0.08)
            ax.set_yticks([])
            ax.set_title(f'Task Number: {i+1}; Language: {lang}')
            ax.set_xlabel(f'Standard Deviation: {embeddings_weights.std().item():.4f}')
            
            count, bins, _ = ax.hist(embeddings_weights.flatten().detach().numpy(), bins=args.bins, density=True)
            ax.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
        
        plt.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.96, hspace=0.75, wspace=0.2)
        if args.save_fn is None:
            args.save_fn = f'appendix_ted_{args.method}.png'

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
        
    save_path = os.path.join(args.save_path, args.save_fn)
    print(f'save figure to {save_path}')
    plt.savefig(save_path, dpi=args.dpi)
