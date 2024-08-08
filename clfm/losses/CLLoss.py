import os
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Iterable, Dict, Tuple

from clfm import CLFramework
from clfm import Constants
from clfm.utils import batch_to_device
from tqdm.autonotebook import tqdm


SUPPORTED_TYPES = [
    'contrastive',
    'mse',
]

class CLLoss(nn.Module):
    def __init__(self, 
                 model: CLFramework, 
                 crosslingual_loss_scale: float,
                 crosslingual_loss_type: str,
                 crossmodal_loss_scale: float,
                 crossmodal_loss_type: str,
                 temperature: float = 0.07,
                 en_loss_scale: float = 0.0,
                 pull_constraint_coeff: float = 0.0,
                 **kwargs,
                 ):
        """
        :param model: CLFramework based on clfm.Framework
        :param crosslingual_loss_scale: teacher's native sentence embeddings <-> student's foreign sentence embeddings
        :param crosslingual_loss_type: either `contrastive` or `mse`
        :param crossmodal_loss_scale: teacher's visual embeddings <-> student's foreign sentence embeddings
        :param crossmodal_loss_type: either `contrastive` or `mse`
        :param en_loss_scale: teacher's native sentence embeddings <-> student's native sentence embeddings
        """
        super(CLLoss, self).__init__()
        self.model = model

        assert all([scale >= 0 for scale in [crosslingual_loss_scale, crossmodal_loss_scale]])
        assert all([loss_type in SUPPORTED_TYPES for loss_type in [crosslingual_loss_type, crossmodal_loss_type]])

        self.crosslingual_loss_scale = crosslingual_loss_scale
        self.crosslingual_loss_type = crosslingual_loss_type

        self.crossmodal_loss_scale = crossmodal_loss_scale
        self.crossmodal_loss_type = crossmodal_loss_type
        
        if self.crosslingual_loss_scale > 0 and self.crosslingual_loss_type == 'contrastive':
            self.tt_temp = nn.Parameter(torch.ones([]) * temperature)

        if self.crossmodal_loss_scale > 0 and self.crossmodal_loss_type == 'contrastive':
            self.vt_temp = nn.Parameter(torch.ones([]) * temperature)
        
        self.en_loss_scale = en_loss_scale

        # for the query-key matching loss of DualPrompt
        self.pull_constraint_coeff = pull_constraint_coeff 
    
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor = None) -> Tuple[Tensor, Dict[str, float]]:
        assert labels is None

        memory_mask = sentence_features['from_memory']
        if memory_mask.sum() == 0:
            # all samples are from the normal dataset
            normal_mask = memory_mask = None
        else:
            normal_mask = 1 - memory_mask

        sentence_features.update(self.model.tokenize(sentence_features['target_texts']))
        sentence_features = batch_to_device(sentence_features, self.model.device)        
        outputs = self.model(sentence_features)

        loss, loss_msg_dict = 0, {}

        if self.crosslingual_loss_scale > 0:
            fct = getattr(self, f'forward_{self.crosslingual_loss_type}')
            this_loss, this_dict = fct(outputs, prefix_name='[TT]_', is_vision_text=False, mask=normal_mask)
            loss += self.crosslingual_loss_scale * this_loss
            loss_msg_dict.update(this_dict)
        
        if self.crossmodal_loss_scale > 0:
            fct = getattr(self, f'forward_{self.crossmodal_loss_type}')
            this_loss, this_dict = fct(outputs, prefix_name='[VT]_', is_vision_text=True, mask=normal_mask)
            loss += self.crossmodal_loss_scale * this_loss
            loss_msg_dict.update(this_dict)
        
        if memory_mask is not None:
            this_loss, this_dict = self.forward_mse(
                outputs=outputs,
                name='loss_replay',
                mask=memory_mask,
            )
            loss += this_loss
            loss_msg_dict.update(this_dict)
        
        if self.pull_constraint_coeff > 0:
            assert 'loss_prompt' in outputs, f"{outputs.keys()}"
            loss += self.pull_constraint_coeff * outputs['loss_prompt']
            loss_msg_dict['loss_prompt'] = outputs['loss_prompt']
        
        if self.en_loss_scale > 0:
            sentence_features.update(self.model.tokenize(sentence_features['source_texts']))
            sentence_features = batch_to_device(sentence_features, self.model.device)        
            outputs = self.model(sentence_features)

            this_loss, this_dict = self.forward_mse(outputs, prefix_name='[En]_', is_vision_text=False, mask=normal_mask)
            loss += self.en_loss_scale * this_loss
            loss_msg_dict.update(this_dict)

        return loss, loss_msg_dict

    def forward_mse(self, 
            outputs: Dict[str, Tensor], 
            name: str = 'loss_mse',
            prefix_name: str = '',
            is_vision_text: bool = False,
            mask: Tensor = None,
            **kwargs,
            ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Computes the MSE loss between the computed sentence embedding and a target sentence embedding. This loss
        is used when extending sentence embeddings to new languages as described in our publication
        Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation: https://arxiv.org/abs/2004.09813

        For an example, see the documentation on extending language models to new languages.
        """
        preds = outputs['sentence_embedding']
        labels = outputs['labels_ve' if is_vision_text else 'labels']
        if mask is None or mask.sum() == len(mask):
            loss_fct = nn.MSELoss()
            loss = loss_fct(preds, labels)
        else:
            assert mask.sum() > 0
            mask = mask.float() / mask.sum()
            loss_fct = nn.MSELoss(reduction='none')
            loss = loss_fct(preds, labels)
            loss = loss.mean([_ for _ in range(1, loss.ndim)])
            loss = (loss * mask).sum()
        return loss, {prefix_name + name: loss.detach().cpu().item()}

    def forward_contrastive(self,
            outputs: Dict[str, Tensor], 
            name: str = 'loss_cl',
            prefix_name: str = '',
            is_vision_text: bool = False,
            **kwargs,
            ) -> Tuple[Tensor, Dict[str, float]]:

        preds = outputs['sentence_embedding']

        if is_vision_text:
            labels = outputs['labels_ve'].to(preds.device)
            temp = self.vt_temp   
            ids = outputs['vision_ids'].view(-1, 1) # (batch_size, 1)
        else:
            labels = outputs['labels']
            temp = self.tt_temp
            ids = None

        with torch.no_grad():
            temp.clamp_(0.001, 0.5)
    
        feats_student = F.normalize(preds, dim=-1)
        feats_teacher = F.normalize(labels, dim=-1)

        logits_s2t = feats_student @ feats_teacher.t() / temp
        logits_t2s = feats_teacher @ feats_student.t() / temp

        if ids is None:
            cl_labels = torch.arange(preds.size(0), device=preds.device)
            loss_s2t = F.cross_entropy(logits_s2t, cl_labels, reduction='mean')
            loss_t2s = F.cross_entropy(logits_t2s, cl_labels, reduction='mean')
        else:
            positive_ids = torch.eq(ids, ids.t()).float() # (batch_size, batch_size)
            cl_labels = positive_ids / positive_ids.sum(1, keepdim=True)
            loss_s2t = - torch.sum(F.log_softmax(logits_s2t, dim=1) * cl_labels, dim=1).mean()
            loss_t2s = - torch.sum(F.log_softmax(logits_t2s, dim=1) * cl_labels, dim=1).mean()

        loss = (loss_s2t + loss_t2s) / 2
        
        return loss, {
            prefix_name + name: loss.detach().cpu().item(),
            prefix_name + 'temp': temp.detach().cpu().item(),
        }

    def before_task(self):
        from ..models.AdapterEncoder import AdapterEncoder
        module: AdapterEncoder = self.model._first_module()
        if hasattr(module, 'prompt') and hasattr(module.prompt, 'process_task_count'):
            print('Process prompt\'s task count ...')
            module.prompt.process_task_count()
        
    def end_task(self, checkpoint_path, *args, **kwargs):
        pass

    @torch.no_grad()
    def test_perturbation(self, train_loader, gaussian_noise_std, use_amp, log_every=100, seed=42, all_parameters=False):
        from zeronlg.utils import seed_everything, MetricLogger
        from ..models.AdapterEncoder import AdapterEncoder

        seed_everything(seed=seed)
        self.model.eval()
        self.model = self.model.to(self.model.device)

        if all_parameters:
            for p in self.model.parameters():
                # noise = torch.from_numpy(np.random.normal(0, gaussian_noise_std, p.shape)).to(p.device)
                noise = torch.randn_like(p) * gaussian_noise_std
                p.add_(noise)
        else:
            module: AdapterEncoder = self.model._first_module()
            assert isinstance(module, AdapterEncoder)

            word_embeddings_weight: nn.Parameter = module.get_word_embeddings().weight
            # noise = torch.from_numpy(np.random.normal(0, gaussian_noise_std, word_embeddings_weight.shape)).to(word_embeddings_weight.device)
            noise = torch.randn_like(word_embeddings_weight) * gaussian_noise_std
            word_embeddings_weight.add_(noise)

        logger = MetricLogger()
        iterator = iter(train_loader)
        for i in tqdm(range(len(train_loader))):
            features, labels = next(iterator)
            if use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    loss, loss_msg_dict  = self.forward(features, labels)
            else:
                loss, loss_msg_dict  = self.forward(features, labels)
            
            logger.update(loss=loss)
            logger.update(**loss_msg_dict)
            if i % log_every == 0:
                pass

        logger.synchronize_between_processes()
        
        info = {}
        for name in logger.meters.keys():
            if 'loss' in name:
                info[name] = logger.meters[name].global_avg
        return info


class oEWC(CLLoss):
    '''
    online EWC (Elastic Weight Consolidation)
    '''
    def __init__(self, 
                 *args, 
                 fisher_penalty_scale: float = 1000, 
                 fisher_gamma: float = 1,
                 **kwargs, 
                 ):
        super().__init__(*args, **kwargs)
        assert fisher_penalty_scale > 0
        assert fisher_gamma > 0
        self.fisher_penalty_scale = fisher_penalty_scale
        self.fisher_gamma = fisher_gamma
        self.fisher = None
        self.old_params = None
    
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor = None) -> Tuple[Tensor, Dict[str, float]]:
        loss_fisher = self.fisher_penalty_scale * self.penalty()
        loss, loss_msg_dict = super().forward(sentence_features, labels)
        loss += loss_fisher
        loss_msg_dict['loss_fisher'] = loss_fisher
        return loss, loss_msg_dict
    
    def penalty(self):
        if self.old_params is None:
            return torch.tensor(0.0).to(self.model.device)
        else:
            penalty = (self.fisher * ((self.get_params() - self.old_params) ** 2)).sum()
            return penalty
    
    def get_params(self, split: bool = False) -> torch.Tensor:
        """
        Returns all trainable parameters concatenated in a single tensor.
        """
        params = []
        embedding_param = None
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                if n.endswith('word_embeddings.weight'):
                    embedding_param = p
                else:
                    params.append(p.view(-1))
        
        if split:
            assert embedding_param is not None
            return torch.cat(params), embedding_param.view(-1)

        if embedding_param is not None:
            params.append(embedding_param.view(-1))

        return torch.cat(params)

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients of trainable parameters concatenated in a single tensor.
        """
        grads = []
        embedding_grad = None
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                if n.endswith('word_embeddings.weight'):
                    embedding_grad = p.grad
                else:
                    grads.append(p.grad.view(-1))
        
        if embedding_grad is not None:
            grads.append(embedding_grad.view(-1))

        return torch.cat(grads)

    def before_task(self):
        super().before_task()
        if self.model.model_name_or_path is not None:
            fisher_path = os.path.join(self.model.model_name_or_path, Constants.FISHER_CKPT_FN)
            if os.path.exists(fisher_path):
                self.fisher = torch.load(fisher_path).to(self.model.device)
                self.old_params = self.get_params().data.clone()
                if len(self.fisher) < len(self.old_params):
                    self.fisher = torch.cat((
                        self.fisher, 
                        self.fisher.new_zeros(len(self.old_params) - len(self.fisher))
                    ))
    
    def end_task(self, train_loader, checkpoint_path, use_amp, scaler, return_fisher_only=False):
        super().end_task(checkpoint_path=checkpoint_path)
        self.model.eval()
        self.model = self.model.to(self.model.device)

        fisher = torch.zeros_like(self.get_params())
        iterator = iter(train_loader)

        for i in tqdm(range(len(train_loader))):
            features, labels = next(iterator)
            self.model.zero_grad()
            if use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    loss, _  = super().forward(features, labels)
                scale = scaler.get_scale()
                (loss * scale).backward()
                this_batch_fisher = (self.get_grads() / scale) ** 2
            else:
                loss, _  = super().forward(features, labels)
                loss.backward()
                this_batch_fisher = self.get_grads() ** 2

            this_batch_fisher[this_batch_fisher.isnan().nonzero().squeeze(1)] = 0.0
            fisher += this_batch_fisher / len(train_loader)

        if self.fisher is None:
            self.fisher = fisher
        else:
            self.fisher *= self.fisher_gamma
            self.fisher += fisher
        
        if return_fisher_only:
            return self.fisher

        fisher_path = os.path.join(checkpoint_path, Constants.FISHER_CKPT_FN)
        torch.save(self.fisher, fisher_path)
