import os
import json
import logging
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, Union, List, Any, Callable
from ..datasets import CaptionDatasetForRetrieval
from .. import Framework, Seq2Seq, LoggingHandler


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
global_logger = logging.getLogger(__name__)


def construct_gts(annotation: List[Dict[str, Any]]):
    img2text = {}
    text2img = {}

    text_id = 0
    for img_id, ann in enumerate(annotation):
        assert 'caption' in ann, f'The annotation item {ann} does not contain the key `caption`'

        captions = ann['caption']
        if not isinstance(captions, (list, tuple)):
            captions = [captions]

        img2text[img_id] = []
        for _ in captions:
            img2text[img_id].append(text_id)
            text2img[text_id] = img_id
            text_id += 1
    
    return img2text, text2img


@torch.no_grad()
def retrieval_evaluation(
        scores_i2t: np.ndarray, 
        scores_t2i: np.ndarray, 
        annotation: List[Dict[str, Any]],
        topk: int = 10,
    ):
    img2text, text2img = construct_gts(annotation)

    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    topk_inds_i2t = np.zeros((scores_i2t.shape[0], topk))
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2text[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        topk_inds_i2t[index] = inds[:topk]

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])
    topk_inds_t2i = np.zeros((scores_t2i.shape[0], topk))
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == text2img[index])[0][0]
        topk_inds_t2i[index] = inds[:topk]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    scores = {
        'txt_r1': tr1,
        'txt_r5': tr5,
        'txt_r10': tr10,
        'txt_r_mean': tr_mean,
        'img_r1': ir1,
        'img_r5': ir5,
        'img_r10': ir10,
        'img_r_mean': ir_mean,
        'r_mean': r_mean
    }
    
    return scores, topk_inds_i2t, topk_inds_t2i


@torch.no_grad()
def retrieval_evaluation_n_fold(
        scores_i2t: np.ndarray, 
        scores_t2i: np.ndarray, 
        annotation: List[Dict[str, Any]],
        n_fold: int = 5,
    ):
    n_fold_annotations = np.array_split(annotation, n_fold)

    all_tr = [[], [], []]
    all_ir = [[], [], []]
    all_mean = [[], [], []]
    i_begin, t_begin = 0, 0
    for i in range(n_fold):
        img2text, text2img = construct_gts(n_fold_annotations[i])

        i_end = i_begin + len(img2text)
        t_end = t_begin + len(text2img)

        # Images->Text
        ranks = np.zeros(len(img2text))
        for index, score in enumerate(scores_i2t[i_begin:i_end, t_begin:t_end]):
            inds = np.argsort(score)[::-1]
            # Score
            rank = 1e20
            for i in img2text[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

        # Compute metrics
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        all_tr[0].append(tr1)
        all_tr[1].append(tr5)
        all_tr[2].append(tr10)

        # Text->Images
        ranks = np.zeros(len(text2img))
        for index, score in enumerate(scores_t2i[t_begin:t_end, i_begin:i_end]):
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == text2img[index])[0][0]

        # Compute metrics
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
        all_ir[0].append(ir1)
        all_ir[1].append(ir5)
        all_ir[2].append(ir10)

        tr_mean = (tr1 + tr5 + tr10) / 3
        ir_mean = (ir1 + ir5 + ir10) / 3
        r_mean = (tr_mean + ir_mean) / 2
        all_mean[0].append(tr_mean)
        all_mean[1].append(ir_mean)
        all_mean[2].append(r_mean)

        i_begin, t_begin = i_end, t_end

    scores = {
        f'{n_fold}fold_txt_r1': np.array(all_tr[0]).mean(),
        f'{n_fold}fold_txt_r5': np.array(all_tr[1]).mean(),
        f'{n_fold}fold_txt_r10': np.array(all_tr[2]).mean(),
        f'{n_fold}fold_txt_r_mean': np.array(all_mean[0]).mean(),
        f'{n_fold}fold_img_r1': np.array(all_ir[0]).mean(),
        f'{n_fold}fold_img_r5': np.array(all_ir[1]).mean(),
        f'{n_fold}fold_img_r10': np.array(all_ir[2]).mean(),
        f'{n_fold}fold_img_r_mean': np.array(all_mean[1]).mean(),
        f'{n_fold}fold_r_mean': np.array(all_mean[2]).mean()
    }
    return scores


class RetrievalEvaluator:
    def __init__(self, 
            loader: Union[DataLoader, List[DataLoader]], 
            mode: str = 'val', 
            logger: logging.Logger = None, 
            monitor: str = 'r_mean',
            with_epoch: bool = False, 
            with_steps: bool = False,
            auto_save: bool = True,
            n_fold: int = 1,
            do_fusion: bool = False,
            fusion_scales: List[float] = None,
            fusion_types: List[str] = None,
            framework_cls: Callable = Framework,
            seq2seq_cls: Callable = Seq2Seq,
        ):
        super().__init__()
        loaders = [loader] if not isinstance(loader, list) else loader

        assert mode in ['val', 'test']
        assert all(isinstance(loader.dataset, CaptionDatasetForRetrieval) for loader in loaders)

        self.loaders = loaders
        self.mode = mode
        self.logger = logger or global_logger
        self.monitor = monitor
        self.with_epoch = with_epoch
        self.with_steps = with_steps
        self.auto_save = auto_save
        self.n_fold = n_fold

        self.do_fusion = do_fusion
        self.fusion_scales = fusion_scales
        self.fusion_types = fusion_types
        if self.do_fusion:
            assert len(self.loaders) == 2, \
                "ensure that there are two loaders for feature or score fusion"
            assert self.fusion_scales is not None
            assert self.fusion_types is not None
        
        self.framework_cls = framework_cls
        self.seq2seq_cls = seq2seq_cls

    def log(self, msg):
        self.logger.info(msg)
    
    def _forward_single_loader(self, model, output_path: str, loader: DataLoader, prefix: str = None):
        if output_path:
            assert prefix is not None
            result_i2t_file = os.path.join(output_path, f'{prefix}_i2t.npy')
            result_t2i_file = os.path.join(output_path, f'{prefix}_t2i.npy')
            scores_file = os.path.join(output_path, f'{prefix}_scores.json')

        if type(model) is str:
            model = self.seq2seq_cls(model)
        elif isinstance(model, self.framework_cls):
            model = self.seq2seq_cls(
                multilingual_model=model,
                device=model.device,
                load_clip_model=False,
            )
        else:
            assert isinstance(model, self.seq2seq_cls)

        all_image_embs, all_text_embs = [], []
        for batch in tqdm(loader):
            _, image_embs, texts = batch
            
            if loader.dataset.clip_model is not None:
                images = None
            else:
                images, image_embs = image_embs, None

            image_embs = model.get_image_embeddings(
                images=images,
                image_embs=image_embs,
                normalize=True,
                mean_pooling=True, # TODO: we now always apply mean pooling
            )

            text_embs = model.get_text_embeddings(
                texts=texts,
                normalize=True,
                batch_size=image_embs.size(0),
            )

            all_image_embs.append(image_embs)
            all_text_embs.append(text_embs)
        
        all_image_embs = torch.cat(all_image_embs, dim=0)
        all_text_embs = torch.cat(all_text_embs, dim=0)

        scores_i2t = all_image_embs @ all_text_embs.t()
        scores_t2i = all_text_embs @ all_image_embs.t()

        scores, topk_inds_i2t, topk_inds_t2i = retrieval_evaluation(
            scores_i2t=scores_i2t.cpu().numpy(),
            scores_t2i=scores_t2i.cpu().numpy(),
            annotation=loader.dataset.annotation,
        )

        if self.n_fold > 1:
            # MSCOCO 1K test
            scores.update(
                retrieval_evaluation_n_fold(
                    scores_i2t=scores_i2t.cpu().numpy(),
                    scores_t2i=scores_t2i.cpu().numpy(),
                    annotation=loader.dataset.annotation,
                    n_fold=self.n_fold,
                )
            )
        
        if output_path:
            self.log(f'Save results to {result_i2t_file}, {result_t2i_file}')
            np.save(result_i2t_file, topk_inds_i2t)
            np.save(result_t2i_file, topk_inds_t2i)

            self.log(f'Save scores to {scores_file}')
            json.dump(scores, open(scores_file, 'w'))
        
        if self.auto_save:
            loader.dataset.save_pickle()

        info = dict(
            all_image_embs=all_image_embs,
            all_text_embs=all_text_embs,
            scores_i2t=scores_i2t,
            scores_t2i=scores_t2i,
        )

        return scores, info

    @torch.no_grad()
    def __call__(self, 
            model: Union[nn.Sequential, nn.Module, str], 
            output_path: str = None, 
            epoch: int = -1, 
            steps: int = -1, 
            fn_prefix: str = None,
            **kwargs,
        ) -> float:

        prefix = [self.mode] if fn_prefix is None else [fn_prefix, self.mode]
        if self.with_epoch:
            prefix.append(f'epoch{epoch}')
        if self.with_steps:
            prefix.append(f'steps{steps}')
        prefix = '_'.join(prefix)

        if len(self.loaders) == 1:
            scores, _ = self._forward_single_loader(model, output_path, self.loaders[0], prefix)
        else:
            merged_scores = {}
            all_info = []
            for loader in self.loaders:
                scores, info = self._forward_single_loader(model, None, loader)
                for k, v in scores.items():
                    merged_scores[k] = merged_scores.get(k, 0) + v
                if self.do_fusion:
                    all_info.append(info)
            
            if self.do_fusion:
                scores = self.fuse(all_info)
            else:
                scores = {k: v / len(self.loaders) for k, v in merged_scores.items()}
            
            if output_path:
                scores_file = os.path.join(output_path, f'{prefix}_scores.json')
                self.log(f'Save scores to {scores_file}')
                json.dump(scores, open(scores_file, 'w'))

        for k, v in scores.items():
            self.log(f'[{prefix}] {k} {v}')

        score = scores.get(self.monitor, 0)
        return score

    def fuse(self, all_info):
        assert len(all_info) == 2

        all_scores = {}
        for fusion_type in self.fusion_types:
            for scale in self.fusion_scales:
                all_image_embs = all_info[0]['all_image_embs']

                if fusion_type == 'feature':
                    all_text_embs = scale * all_info[0]['all_text_embs'] \
                        + (1 - scale) * all_info[1]['all_text_embs']
                    all_text_embs = F.normalize(all_text_embs, dim=-1)
                    scores_i2t = all_image_embs @ all_text_embs.t()
                    scores_t2i = all_text_embs @ all_image_embs.t()
                else:
                    scores_i2t = scale * all_info[0]['scores_i2t'] + (1 - scale) * all_info[1]['scores_i2t']
                    scores_t2i = scale * all_info[0]['scores_t2i'] + (1 - scale) * all_info[1]['scores_t2i']
                
                scores, topk_inds_i2t, topk_inds_t2i = retrieval_evaluation(
                    scores_i2t=scores_i2t.cpu().numpy(),
                    scores_t2i=scores_t2i.cpu().numpy(),
                    annotation=self.loaders[0].dataset.annotation,
                )
                scores = {"%s_%.2f_%s" % (fusion_type, scale, k): v for k, v in scores.items()}
                all_scores.update(scores)

        return all_scores
    
