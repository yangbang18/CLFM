import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Union, List, Optional
import matplotlib.pyplot as plt


def gram_schmidt(vv: torch.Tensor, num_new_tensors=1, identical_distribution=False) -> torch.Tensor:
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    # swap rows and columns
    vv = vv.T
    uu = vv.new_zeros(vv.size(0), vv.size(1) + num_new_tensors)
    uu[:, :-num_new_tensors] = vv.clone()

    for k in range(vv.size(1), vv.size(1) + num_new_tensors):
        vk = torch.randn_like(vv[:, 0]).to(vv.device)
        uk = 0
        for j in range(0, k):
            uj = uu[:, j].clone()
            proj = projection(uj, vk)
            uk = uk + proj
        uu[:, k] = vk - uk
        uk = uu[:, k].clone()
        if identical_distribution:
            uu[:, k] = (uk - uk.mean()) / uk.std() * uu[:, :k].std() + uu[:, :k].mean()
        else:
            # orthonormal
            uu[:, k] = uk / (uk.norm())
    
    return uu[:, -num_new_tensors:].T


def gram_schmidt2(vv, start, end):
    def projection(u, v):
        denominator = (u * u).sum()

        if denominator < 1e-8:
            return None
        else:
            return (v * u).sum() / denominator * u
    
    # swap rows and columns
    vv = vv.T

    uu = torch.zeros_like(vv, device=vv.device)
    if start > 0:
        uu[:, 0:start] = vv[:, 0:start].clone()
    
    for k in range(start, end):
        redo = True
        while redo:
            redo = False
            vk = torch.randn_like(vv[:,k]).to(vv.device)
            uk = 0
            for j in range(0, k):
                if not redo:
                    uj = uu[:, j].clone()
                    proj = projection(uj, vk)
                    if proj is None:
                        redo = True
                        print('restarting!!!')
                    else:
                        uk = uk + proj
            if not redo: 
                uu[:, k] = vk - uk
    
    for k in range(start, end):
        uk = uu[:, k].clone()
        uu[:, k] = uk / (uk.norm())

    # undo swapping of rows and columns
    uu = uu.T
    return uu


class TokenModel(nn.Module):
    def __init__(
        self, 
        old_bos_token_ids, 
        new_bos_token_ids, 
        embeddings, 
        old_vocab_size, 

        reduce_simi_of_old_bos_and_new_tokens: bool = False,
        increase_simi_of_new_bos_and_new_tokens: bool = False,
        margin: float = 0.1,
        old_mode: str = 'mean', 
        new_mode: str ='mean',

        new_old_bos_most_similar: bool = False,
        new_old_bos_similarity: Optional[Union[float, List[float]]] = None,
    ):
        super().__init__()
        if reduce_simi_of_old_bos_and_new_tokens:
            self.new_token_embeddings = nn.Parameter(embeddings[old_vocab_size:, :].clone())

        if increase_simi_of_new_bos_and_new_tokens or new_old_bos_most_similar or new_old_bos_similarity:
            self.new_bos_token_embeddings = nn.Parameter(embeddings[new_bos_token_ids, :].clone())
        
        self.normalized_embeddings = F.normalize(embeddings, dim=-1)
        
        self.old_bos_token_ids = old_bos_token_ids
        self.new_bos_token_ids = new_bos_token_ids
        self.old_vocab_size = old_vocab_size

        self.reduce_simi_of_old_bos_and_new_tokens = reduce_simi_of_old_bos_and_new_tokens
        self.increase_simi_of_new_bos_and_new_tokens = increase_simi_of_new_bos_and_new_tokens
        self.margin = margin
        self.old_mode = old_mode
        self.new_mode = new_mode

        self.new_old_bos_most_similar = new_old_bos_most_similar
        self.new_old_bos_similarity = new_old_bos_similarity

    def pool(self, tensor, mode):
        if mode == 'min':
            return tensor.min(dim=-1)[0]
        elif mode == 'max':
            return tensor.max(dim=-1)[0]
        return tensor.mean(dim=-1)

    def forward(self):
        loss = 0

        if self.reduce_simi_of_old_bos_and_new_tokens:
            old_bos_to_new_tokens = self.normalized_embeddings[self.old_bos_token_ids, :] @ F.normalize(self.new_token_embeddings, dim=-1).T
            old_bos_to_new_tokens = self.pool(old_bos_to_new_tokens, self.new_mode)
            old_bos_to_old_tokens = self.normalized_embeddings[self.old_bos_token_ids, :] @ self.normalized_embeddings[:self.old_vocab_size, :].T
            old_bos_to_old_tokens = self.pool(old_bos_to_old_tokens, self.old_mode)
            loss += (old_bos_to_new_tokens + self.margin - old_bos_to_old_tokens).clamp(min=0).mean()

        if self.increase_simi_of_new_bos_and_new_tokens or self.new_old_bos_most_similar or self.new_old_bos_similarity is not None:
            simi = F.normalize(self.new_bos_token_embeddings, dim=-1) @ self.normalized_embeddings.T

        if self.increase_simi_of_new_bos_and_new_tokens:
            new_bos_to_old_tokens = self.pool(simi[:, :self.old_vocab_size], self.old_mode)
            new_bos_to_new_tokens = self.pool(simi[:, self.old_vocab_size:], self.new_mode)
            loss += (new_bos_to_old_tokens + self.margin - new_bos_to_new_tokens).clamp(min=0).mean()

        if self.new_old_bos_most_similar:
            simi_to_old_bos = simi[:, self.old_bos_token_ids]
            inf_mask = simi.new_zeros(*simi.shape)
            inf_mask[:, self.old_bos_token_ids + self.new_bos_token_ids] = -1e4
            maximun_simi_to_others = (simi + inf_mask).max(dim=1)[0]
            loss += (maximun_simi_to_others - simi_to_old_bos).clamp(min=0).mean()
            self.simi_to_old_bos = simi_to_old_bos

        if self.new_old_bos_similarity is not None:
            if isinstance(self.new_old_bos_similarity, list):
                lower_bound, upper_bound = self.new_old_bos_similarity
            else:
                lower_bound = upper_bound = self.new_old_bos_similarity
            simi_to_old_bos = simi[:, self.old_bos_token_ids]
            loss += (lower_bound - simi_to_old_bos).clamp(min=0).mean()
            loss += (simi_to_old_bos - upper_bound).clamp(min=0).mean()
            self.simi_to_old_bos = simi_to_old_bos

        return loss


def probe(
        token_ids: Union[List[int], int], 
        embeddings: torch.Tensor, 
        base_tokens_end_position: int, 
        new_tokens_start_position: Optional[int]=None, 
        tokenizer=None, 
        labels: List[str]=['old tokens', 'new tokens'], 
        bins: int=400, 
        figsize=(5, 2.5), 
        title=None,
        output_path: Optional[str]=None,
        only_new=False,
        add_mean=True,
    ):
    if title is None:
        assert tokenizer is not None
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        title = f'similarities between {tokens} and the others'

    if only_new:
        assert isinstance(token_ids, list)
        assert len(token_ids) == len(labels)
    else:
        assert len(labels) == 2

    if not isinstance(token_ids, list):
        token_ids = [token_ids]

    new_tokens_start_position = new_tokens_start_position or base_tokens_end_position

    fig = plt.figure(figsize=figsize)
    normalized_embeddings = F.normalize(embeddings.float(), dim=-1)
    
    simi = normalized_embeddings[token_ids] @ normalized_embeddings.T
    ax = plt.subplot(1, 1, 1)
    if only_new:
        simi = simi.numpy()
        for this_simi, label in zip(simi, labels):
            _ = ax.hist(
                this_simi[new_tokens_start_position:], 
                bins=int(bins), 
                alpha=0.4,
                density=True, 
                label=label + (' (mean=%.4f)' % this_simi[new_tokens_start_position:].mean() if add_mean else ''),
            )
    else:
        simi = simi.mean(0).numpy()
        
        _ = ax.hist(
            simi[:base_tokens_end_position], 
            bins=int(bins), 
            alpha=0.4,
            density=True, 
            label=labels[0] + (' (mean=%.4f)' % simi[:base_tokens_end_position].mean() if add_mean else ''),
        )
        _ = ax.hist(
            simi[new_tokens_start_position:], 
            bins=int(bins), 
            alpha=0.4,
            density=True, 
            label=labels[1] + (' (mean=%.4f)' % simi[new_tokens_start_position:].mean() if add_mean else ''),
        )
    ax.set_ylabel('Density')
    ax.set_xlabel('Cosine Similarity')
    ax.set_title(title)
    plt.legend()
    plt.subplots_adjust(right=0.99, left=0.12, bottom=0.20, top=0.86)
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


def plot(emb1, emb2, bins=50, fontsize=16, figsize=(9, 2.7), output_path=None):
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1)
    a = ax.hist(
        emb1.squeeze().cpu().numpy(),
        bins=int(bins),
        alpha=0.4,
        density=True,
        label='before SGD',
    )
    b = ax.hist(
        emb2.squeeze().cpu().numpy(),
        bins=int(bins),
        alpha=0.4,
        density=True,
        color='green',
        label='after SGD',
    )
    plt.legend()
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


def sgd_init(
    embeddings: torch.Tensor,
    old_vocab_size: int,
    new_bos_token_ids: Union[int, List[int]],
    old_bos_token_ids: Union[int, List[int]],
    tokenizer,

    reduce_simi_of_old_bos_and_new_tokens: bool = False,
    increase_simi_of_new_bos_and_new_tokens: bool = False,
    margin: float = 0.1,
    old_mode: str = 'mean', 
    new_mode: str ='mean',

    new_old_bos_most_similar: bool = False,
    new_old_bos_similarity: Optional[Union[float, List[float]]] = None,
    
    steps: int = 1000,
    log_steps: int = 50,
    lr: float = 0.1,
    output_path: Optional[str] = None,

    truncate_values: bool = False,
) -> torch.Tensor:
    """
        rule1: new and old bos tokens should be the most similar
    """
    if isinstance(new_bos_token_ids, list):
        assert len(new_bos_token_ids) == 1, 'we only support learning a new language for a new task'
    
    if not isinstance(old_bos_token_ids, list):
        old_bos_token_ids = [old_bos_token_ids]


    assert new_old_bos_similarity is not None or margin is not None

    model = TokenModel(
        old_bos_token_ids=old_bos_token_ids,
        new_bos_token_ids=new_bos_token_ids,
        embeddings=embeddings,
        old_vocab_size=old_vocab_size,

        reduce_simi_of_old_bos_and_new_tokens=reduce_simi_of_old_bos_and_new_tokens,
        increase_simi_of_new_bos_and_new_tokens=increase_simi_of_new_bos_and_new_tokens,
        margin=margin,
        old_mode=old_mode,
        new_mode=new_mode,

        new_old_bos_most_similar=new_old_bos_most_similar,
        new_old_bos_similarity=new_old_bos_similarity,
    ).train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    total_loss = 0
    for i in range(steps):
        model.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_steps == 0:
            logging.info(f'Step {i:03d}:\tloss={loss.item():.4f}\tavg loss={total_loss / (i+1)}')
        
        if truncate_values:
            if hasattr(model, 'new_bos_token_embeddings'):
                mean, std = model.new_bos_token_embeddings.data.mean().item(), model.new_bos_token_embeddings.data.std().item()
                model.new_bos_token_embeddings.data.clamp_(min=mean - 3 * std, max=mean + 3 * std)
            
            if hasattr(model, 'new_token_embeddings'):
                mean, std = model.new_token_embeddings.data.mean().item(), model.new_token_embeddings.data.std().item()
                model.new_token_embeddings.data.clamp_(min=mean - 3 * std, max=mean + 3 * std)
    
    emb1 = embeddings[new_bos_token_ids].clone()
    
    if hasattr(model, 'new_token_embeddings'):
        if output_path is not None:
            probe(old_bos_token_ids, embeddings, old_vocab_size, tokenizer=tokenizer, output_path=os.path.join(output_path, 'simi_of_old_bos_and_tokens_before_sgd.png'))
        
        embeddings[old_vocab_size:] = model.new_token_embeddings.data

        if output_path is not None:
            probe(old_bos_token_ids, embeddings, old_vocab_size, tokenizer=tokenizer, output_path=os.path.join(output_path, 'simi_of_old_bos_and_tokens_after_sgd.png'))

        logging.info(f'Statistics of new tokens: mean={embeddings[old_vocab_size:].mean().item():.4f}, std={embeddings[old_vocab_size:].std().item():.4f}')

    if hasattr(model, 'new_bos_token_embeddings'):
        if output_path is not None:
            probe(new_bos_token_ids, embeddings, old_vocab_size, tokenizer=tokenizer, output_path=os.path.join(output_path, 'simi_of_new_bos_and_tokens_before_sgd.png'))
        
        embeddings[new_bos_token_ids] = model.new_bos_token_embeddings.data

        if output_path is not None:
            probe(new_bos_token_ids, embeddings, old_vocab_size, tokenizer=tokenizer, output_path=os.path.join(output_path, 'simi_of_new_bos_and_tokens_after_sgd.png'))

        logging.info(f'Statistics of new bos tokens: mean={embeddings[new_bos_token_ids].mean().item():.4f}, std={embeddings[new_bos_token_ids].std().item():.4f}')
    
    if hasattr(model, 'simi_to_old_bos'):
        logging.info(f'simi_to_old_bos: {model.simi_to_old_bos}')
    
    emb2 = embeddings[new_bos_token_ids].clone()
    if output_path is not None:
        plot(emb1, emb2, output_path=os.path.join(output_path, 'bos_distribution.png'))

    return embeddings


def repeat_init(
    embeddings: torch.Tensor,
    base_bos_token_ids: List[int],
    new_bos_token_ids: List[int], 
    base_tokens_end_position: int,
    new_tokens_start_position: int,
    mean: float = 0,
    std: float = 0.02,
    reduce_simi_of_base_bos_and_new_tokens: bool = False,
    increase_simi_of_new_bos_and_new_tokens: bool = False,
    ensure_max_simi_of_new_bos_and_new_tokens: bool = False,
    repeat_times: int = 1000,
):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    normalized_embeddings = F.normalize(embeddings, dim=-1).to(device)
    normalized_base_bos_embeds = normalized_embeddings[base_bos_token_ids]
    normalized_base_tokens_embeds = normalized_embeddings[:base_tokens_end_position]
    normalized_new_tokens_embeds = normalized_embeddings[new_tokens_start_position:]

    num_new_tokens = embeddings.size(0) - new_tokens_start_position
    if reduce_simi_of_base_bos_and_new_tokens:
        new_tokens_embeds = torch.zeros(repeat_times * num_new_tokens, embeddings.size(1)).to(device)
        new_tokens_embeds.normal_(mean=mean, std=std)

        simi_of_base_bos_and_new_tokens = normalized_base_bos_embeds @ F.normalize(new_tokens_embeds, dim=-1).T
        simi_of_base_bos_and_new_tokens = simi_of_base_bos_and_new_tokens.view(len(base_bos_token_ids), repeat_times, num_new_tokens).mean(dim=(0, 2))

        simi_of_base_bos_and_base_toknes = (normalized_base_bos_embeds @ normalized_base_tokens_embeds.T).mean()

        diff = simi_of_base_bos_and_base_toknes - simi_of_base_bos_and_new_tokens
        best_candidate_index = diff.argmax()
        best_candidate = new_tokens_embeds.view(repeat_times, num_new_tokens, embeddings.size(1))[best_candidate_index]

        embeddings[new_tokens_start_position:] = best_candidate.to(embeddings.device)
        normalized_new_tokens_embeds = F.normalize(best_candidate, dim=-1)
        print('base diff:', diff[best_candidate_index])
    
    if increase_simi_of_new_bos_and_new_tokens:
        new_bos_embeds = torch.zeros(repeat_times * len(new_bos_token_ids), embeddings.size(1)).to(device)
        new_bos_embeds.normal_(mean=mean, std=std)
        normalized_new_bos_embeds = F.normalize(new_bos_embeds, dim=-1)

        simi_of_new_bos_and_base_tokens = normalized_new_bos_embeds @ normalized_base_tokens_embeds.T
        simi_of_new_bos_and_base_tokens = simi_of_new_bos_and_base_tokens.view(repeat_times, len(new_bos_token_ids), -1).mean(dim=(1, 2))

        simi_of_new_bos_and_new_tokens = normalized_new_bos_embeds @ normalized_new_tokens_embeds.T
        simi_of_new_bos_and_new_tokens = simi_of_new_bos_and_new_tokens.view(repeat_times, len(new_bos_token_ids), -1).mean(dim=(1, 2))
        
        diff = simi_of_new_bos_and_new_tokens - simi_of_new_bos_and_base_tokens
        best_candidate_index = diff.argmax()
        best_candidate = new_bos_embeds.view(repeat_times, len(new_bos_token_ids), embeddings.size(1))[best_candidate_index]
        embeddings[new_bos_token_ids] = best_candidate.to(embeddings.device)
        print('new diff:', diff[best_candidate_index])
    
    if ensure_max_simi_of_new_bos_and_new_tokens:
        max_simi_of_base_bos_and_new_tokens = (normalized_base_bos_embeds @ normalized_new_tokens_embeds.T).mean(dim=1).max()
        
        new_bos_embeds = torch.zeros(repeat_times * len(new_bos_token_ids), embeddings.size(1)).to(device)
        new_bos_embeds.normal_(mean=mean, std=std)
        normalized_new_bos_embeds = F.normalize(new_bos_embeds, dim=-1)
        
        simi_of_new_bos_and_new_tokens = normalized_new_bos_embeds @ normalized_new_tokens_embeds.T
        simi_of_new_bos_and_new_tokens = simi_of_new_bos_and_new_tokens.view(repeat_times, len(new_bos_token_ids), -1).mean(dim=(1, 2))
        
        diff = simi_of_new_bos_and_new_tokens - max_simi_of_base_bos_and_new_tokens
        best_candidate_index = diff.argmax()
        best_candidate = new_bos_embeds.view(repeat_times, len(new_bos_token_ids), embeddings.size(1))[best_candidate_index]
        embeddings[new_bos_token_ids] = best_candidate.to(embeddings.device)
        print('new diff:', diff[best_candidate_index])

    return embeddings
