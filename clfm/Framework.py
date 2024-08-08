import os
import torch
import json
import time
import logging
import datetime
import transformers
import torch
import numpy as np
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm.autonotebook import trange
from typing import List, Dict, Tuple, Iterable, Type, Callable, Optional, Union, Any
from collections import defaultdict

import sentence_transformers
import zeronlg
from zeronlg.models import Projector, Decoder

from .utils import (
    get_cache_folder, 
    seed_everything,
    MetricLogger,
    download_if_necessary,
    batch_to_device,
)
from .optimizers import AdamW
from .models import CLIPModel, MCLIP, AdapterEncoder, SBERT_MAPPINGS
from .Constants import CLIP_MODELS, MCLIP_MODELS
from . import __LIBRARY_NAME__, __version__, __HUGGINGFACE_HUB_NAME__


class Framework(zeronlg.Framework):
    def __init__(self, 
                 model_name_or_path: Optional[str] = None, 
                 modules: Optional[Iterable[nn.Module]] = None, 
                 device: Optional[str] = None, 
                 cache_folder: Optional[str] = get_cache_folder(), 
                 use_auth_token: Union[bool, str, None] = None,
                 tie_word_embeddings: bool = True,
                 tie_all: bool = True,
                 init_word_embeddings: bool = False,
                 freeze_word_embeddings: bool = False,
                 logger: logging.Logger = None,
                 load_sbert_only: bool = False,
                 add_version: bool = True,
                 ):

        self.cache_folder = cache_folder
        self.sbert_mappings = SBERT_MAPPINGS

        if model_name_or_path in CLIP_MODELS + MCLIP_MODELS:
            model_name_or_path = download_if_necessary(model_name_or_path, cache_folder, use_auth_token)
        self.model_name_or_path = model_name_or_path

        super().__init__(
            model_name_or_path, 
            modules=modules, 
            device=device, 
            cache_folder=cache_folder, 
            use_auth_token=use_auth_token,
            tie_word_embeddings=tie_word_embeddings,
            tie_all=tie_all,
            init_word_embeddings=init_word_embeddings,
            freeze_word_embeddings=freeze_word_embeddings,
            logger=logger,
            load_sbert_only=load_sbert_only,
            add_version=False,
        )

        if add_version:
            if '__version__' not in self._model_config:
                self._model_config['__version__'] = {
                    'sentence_transformers': sentence_transformers.__version__,
                    'transformers': transformers.__version__,
                    'pytorch': torch.__version__,
                    __LIBRARY_NAME__: __version__,
                }
            elif __LIBRARY_NAME__ not in self._model_config['__version__']:
                self._model_config['__version__'][__LIBRARY_NAME__] = __version__
    
    def forward(self, features):
        module = self._first_module()
        if isinstance(module, AdapterEncoder):
            prompt = getattr(module, 'prompt', None)
            if prompt is not None and 'cls' in getattr(prompt, 'embedding_key', 'cls'):
                # get cls features, which will be used as the query to get relevant prompts of each sample
                # use current multilingual model as the query network
                is_train = self.training
                self.eval()
                features['with_prompt'] = False
                with torch.no_grad():
                    features['cls_features'] = super().forward(features)['sentence_embedding']
                features['with_prompt'] = True
                self.train(is_train)

        return super().forward(features)
    
    def get_embeddings(self):
        for module in self.get_modules():
            if hasattr(module, 'get_embeddings'):
                return module.get_embeddings()

    def get_input_embeddings(self):
        for module in self.get_modules():
            if hasattr(module, 'get_input_embeddings'):
                return module.get_input_embeddings()
            if hasattr(module, 'get_word_embeddings'):
                return module.get_word_embeddings()
            if hasattr(module, 'auto_model'):
                return module.auto_model.get_input_embeddings()

    def _get_specific_model(self, before=True, instances=(Projector, Decoder), device=None, return_modules_only: bool = False, **kwargs):
        """only keep related modules"""
        modules = self.get_modules()
        idx = 0
        for module in modules:
            if isinstance(module, instances):
                break
            idx += 1

        device = device or self.device

        if before:
            # get modules < idx
            if idx == 0:
                return None
            if return_modules_only:
                return modules[:idx]
            model = Framework(modules=modules[:idx], device=device, **kwargs)
        else:
            # get modules >= idx
            if idx == len(modules):
                return None
            if return_modules_only:
                return modules[idx:]
            model = Framework(modules=modules[idx:], device=device, **kwargs)

        model.to(device)
        return model

    def _load_sbert_model(self, model_path):
        """
        Loads a full sentence-transformers model
        """

        # check version before loading
        config_sentence_transformers_json_path = os.path.join(model_path, 'config_sentence_transformers.json')

        if os.path.exists(config_sentence_transformers_json_path):
            with open(config_sentence_transformers_json_path) as fIn:
                config = json.load(fIn)

            for package_name, version in zip(
                ['sentence_transformers', 'zeronlg', __LIBRARY_NAME__], 
                [sentence_transformers.__version__, zeronlg.__version__, __version__]
            ):
                if '__version__' in config \
                    and package_name in config['__version__'] \
                    and config['__version__'][package_name] != version:
                    self.logger.warning(
                        f"You try to use a {package_name} model that was created with version {config['__version__'][package_name]}, however, your version is {version}. \
                        This might cause unexpected behavior or errors.\n\n\n")
        
        return super()._load_sbert_model(model_path)

    def _load_auto_model(self, model_path):
        """
        For CLIP-like models, return it as the module;
        Otherwise, return modules [Transformer, Mean Pooling].
        """
        # check if model_path refers to a clip-like model
        if any([name.replace('/', '_') in model_path for name in CLIP_MODELS]):
            return [CLIPModel(model_path)]
        elif any([name.replace('/', '_') in model_path for name in MCLIP_MODELS]):
            return [MCLIP(model_path)]

        return super()._load_auto_model(model_path)

    def call_module_function(self, function_name, *args, **kwargs):
        for module in self.get_modules():
            if hasattr(module, function_name):
                getattr(module, function_name)(*args, **kwargs)

    def set_module_attribute(self, module_class, key, value):
        for module in self.get_modules():
            if isinstance(module, module_class):
                setattr(module, key, value)
    
    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator: zeronlg.CaptionEvaluator = None,
            epochs: int = 1,
            steps_per_epoch = None,
            scheduler: str = 'WarmupLinear',
            no_decay: List[str] = ['bias', 'LayerNorm.bias', 'LayerNorm.weight'],
            high_lr_keys: List[str] = [],
            high_lr_factor: float = 100,
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = torch.optim.AdamW,
            optimizer_params : Dict[str, object]= {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True,
            checkpoint_path: str = None,
            checkpoint_save_steps: int = 500,
            checkpoint_save_total_limit: int = 0,
            log_every: int = 500,
            seed: int = 42,
            use_masking: bool = False,
            mask_prob: float = 0.15,
            writter: SummaryWriter = None,
            skip_training: bool = False,
            **kwargs,
            ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size from train_objectives.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        :param checkpoint_path: Folder to save checkpoints during training
        :param checkpoint_save_steps: Will save a checkpoint after so many steps
        :param checkpoint_save_total_limit: Total number of checkpoints to store
        """
        seed_everything(seed=seed)

        self.use_masking = use_masking
        self.mask_prob = mask_prob

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.to(self.device)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self.device)

        self.best_score = -9999999
        self.best_epoch = None
        self.best_steps = None

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            optimizer = self._get_optimizer(loss_model, optimizer_class, optimizer_params, weight_decay, no_decay, 
                                            high_lr_keys=high_lr_keys, high_lr_factor=high_lr_factor, 
                                            steps_per_epoch=steps_per_epoch, num_train_steps=num_train_steps)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)
            
            if hasattr(loss_model, 'before_task'):
                loss_model.before_task()

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        train_start_time = time.time()
        if not skip_training:
            for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
                self._training_epoch_start(epoch, optimizers)

                training_steps = 0
                metric_logger = MetricLogger(delimiter="  ")
                start_time = time.time()

                for loss_model in loss_models:
                    loss_model.zero_grad()
                    loss_model.train()

                for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                    for train_idx in range(num_train_objectives):
                        loss_model = loss_models[train_idx]
                        optimizer = optimizers[train_idx]
                        scheduler = schedulers[train_idx]
                        data_iterator = data_iterators[train_idx]
                        optimizer.zero_grad()
                        
                        self._training_step_start(training_steps, global_step, epoch, loss_model, optimizer)

                        try:
                            data = next(data_iterator)
                        except StopIteration:
                            data_iterator = iter(dataloaders[train_idx])
                            data_iterators[train_idx] = data_iterator
                            data = next(data_iterator)

                        features, labels = data
                        labels = labels.to(self.device) if labels is not None else None
                        features = batch_to_device(features, self.device)
                        features['epoch'] = epoch
                        features['global_step'] = global_step
                        features['num_train_steps'] = num_train_steps

                        if use_amp:
                            with autocast():
                                loss_value, loss_msg_dict = loss_model(features, labels)

                            scale_before_step = scaler.get_scale()
                            scaler.scale(loss_value).backward()
                            scaler.unscale_(optimizer)
                            if max_grad_norm is not None and max_grad_norm > 0:
                                torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                            scaler.step(optimizer)
                            scaler.update()
                            skip_scheduler = scaler.get_scale() != scale_before_step
                        else:
                            loss_value, loss_msg_dict = loss_model(features, labels)
                            loss_value.backward()
                            if max_grad_norm is not None and max_grad_norm > 0:
                                torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                            optimizer.step()
                        
                        self._training_step_end(features, dataloaders[train_idx], is_the_first_epoch=epoch==0)

                        metric_logger.update(**loss_msg_dict)
                        
                        if not skip_scheduler:
                            scheduler.step()

                    training_steps += 1
                    global_step += 1

                    if log_every > 0 and global_step % log_every == 0:
                        self.log_training_info(metric_logger, epoch, training_steps, steps_per_epoch, writter=writter)

                    if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                        self._before_evaluation(loss_models)
                        self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback, writter=writter)
                        self._after_evaluation(loss_models)

                        for loss_model in loss_models:
                            loss_model.zero_grad()
                            loss_model.train()

                        info = f"[BEST] {self.best_score}"
                        self.logger.info(info)

                    if checkpoint_path is not None and checkpoint_save_steps is not None and checkpoint_save_steps > 0 and global_step % checkpoint_save_steps == 0:
                        self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)
                
                metric_logger.synchronize_between_processes()
                info = f"Averaged stats: {metric_logger.global_avg()}"
                self.logger.info(info)
                time_string = 'Train epoch time: ' + str(datetime.timedelta(seconds=int(time.time() - start_time)))
                self.logger.info(time_string)
                if writter is not None:
                    for k, v in metric_logger.meters.items():
                        writter.add_scalar(k + '/epoch', v.global_avg, global_step=epoch)

                self._before_evaluation(loss_models)
                self._eval_during_training(evaluator, output_path, save_best_model, epoch, global_step, callback, writter=writter)
                self._after_evaluation(loss_models)

                if evaluator is None and checkpoint_path is not None and checkpoint_save_steps is None:
                    self._save_checkpoint_epoch(checkpoint_path, checkpoint_save_total_limit, epoch)
                
                self._training_epoch_end(epoch)

        if (evaluator is None and output_path is not None) or skip_training:   #No evaluator, but output path: save final model version
            self.save(output_path)

        if checkpoint_path is not None and checkpoint_save_steps is not None:
            self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)
        
        time_string = 'Train time: ' + str(datetime.timedelta(seconds=int(time.time() - train_start_time)))
        self.logger.info(time_string)

        for j, loss_model in enumerate(loss_models):
            if hasattr(loss_model, 'end_task'):
                start_time = time.time()
                loss_model.end_task(
                    train_loader=dataloaders[j], 
                    checkpoint_path=checkpoint_path, 
                    use_amp=use_amp, 
                    scaler=scaler,
                )
                time_string = 'End task time: ' + str(datetime.timedelta(seconds=int(time.time() - start_time)))
                self.logger.info(time_string)

    def _training_step_start(self, training_steps: int, global_step: int, epoch: int, loss_model: nn.Module, optimizer):
        return

    def _training_step_end(self, features: Dict[str, Any], dataloader: DataLoader, is_the_first_epoch: bool):
        return
    
    def _training_epoch_end(self, epoch: int):
        return
    
    def _training_epoch_start(self, epoch: int, optimizers):
        return
    
    def _before_evaluation(self, loss_models: List[nn.Module]):
        for loss_model in loss_models:
            if hasattr(loss_model, 'before_evaluation'):
                loss_model.before_evaluation()
    
    def _after_evaluation(self, loss_models: List[nn.Module]):
        for loss_model in loss_models:
            if hasattr(loss_model, 'after_evaluation'):
                loss_model.after_evaluation()

    def _get_optimizer(self, loss_model, optimizer_class, optimizer_params, weight_decay, no_decay, high_lr_keys=[], high_lr_factor=100, **kwargs):
        lr = optimizer_params.pop('lr', None)
        assert lr is not None

        no_decay_parameter_list = []
        decay_parameter_list = []
        no_decay_parameter_list_high = []
        decay_parameter_list_high = []

        for n, p in loss_model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or any(nd in n for nd in no_decay):
                if any(key in n for key in high_lr_keys):
                    no_decay_parameter_list_high.append(p)
                else:
                    no_decay_parameter_list.append(p)
            else:
                if any(key in n for key in high_lr_keys):
                    decay_parameter_list_high.append(p)
                else:
                    decay_parameter_list.append(p)

        optimizer_grouped_parameters = [
            {'params': no_decay_parameter_list, 'weight_decay': 0.0, 'lr': lr},
            {'params': decay_parameter_list, 'weight_decay': weight_decay, 'lr': lr},
            {'params': no_decay_parameter_list_high, 'weight_decay': 0.0, 'lr': lr * high_lr_factor},
            {'params': decay_parameter_list_high, 'weight_decay': weight_decay, 'lr': lr * high_lr_factor},
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        return optimizer

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback, writter: SummaryWriter=None):
        """Runs evaluation during the training"""
        eval_path = output_path
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            eval_path = os.path.join(output_path, "eval")
            os.makedirs(eval_path, exist_ok=True)

        if evaluator is not None:
            score = evaluator(self, output_path=eval_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                self.best_epoch = epoch
                self.best_steps = steps
                if save_best_model:
                    self.save(output_path)
            self.logger.info(f'[BEST] score: {self.best_score}, epoch: {self.best_epoch}, steps: {self.best_steps}')

            if writter is not None:
                writter.add_scalar('BEST/step', self.best_score, global_step=steps)
                writter.add_scalar('BEST/epoch', self.best_score, global_step=epoch)
    
    def log_training_info(self, 
            metric_logger: MetricLogger, 
            epoch: int, 
            step: int, 
            steps_per_epoch: int,
            delimiter: str = '  ',
            writter: SummaryWriter=None,
        ):
        super().log_training_info(metric_logger, epoch, step, steps_per_epoch, delimiter)
        if writter is not None:
            for k, v in metric_logger.meters.items():
                writter.add_scalar(k + '/step', v.global_avg, global_step=step)
    
    @staticmethod
    def load(input_path):
        return Framework(input_path)
    

class CLFramework(Framework):
    @staticmethod
    def smart_batching_collate(batch):
        """Transforms a batch of InputExample to features requested by this model"""
        def to_tensor(data, dtype=torch.float32):
            return None if data[0] is None else torch.tensor(np.array(data)).to(dtype)

        features = {}
        features['examples'] = [example for example in batch]
        features['source_texts'] = [example.src_text for example in batch]
        features['target_texts'] = [example.trg_text for example in batch]
        features['labels'] = to_tensor([example.label for example in batch], torch.float32)
        features['labels_ve'] = to_tensor([example.label_ve for example in batch], torch.float32)
        features['vision_ids'] = to_tensor([example.sid for example in batch], torch.int64)
        features['from_memory'] = to_tensor([int(example.from_memory) for example in batch], torch.float32)
        return features, None

    def _training_step_end(self, features: Dict[str, Any], dataloader: DataLoader, is_the_first_epoch: bool):
        from .datasets import ConcatDataset
        dataset = dataloader.dataset
        if isinstance(dataset, ConcatDataset) and dataset.experience_replay:
            dataset.update_memory(examples=features['examples'], labels=features['sentence_embedding'])

    def _get_optimizer(self, loss_model, optimizer_class, optimizer_params, weight_decay, no_decay, high_lr_keys=[], high_lr_factor=100, **kwargs):
        if hasattr(self, 'regularize') and self.regularize:
            assert hasattr(self, 'regularize_type')
            assert hasattr(self, 'grad_scale_type')
            assert hasattr(self, 'weight_decay_scale_type')
            assert hasattr(self, 'exclude_special_tokens')

            embedding_keys = ['word_embeddings']
            embedding_parameter_list = []
            no_decay_parameter_list = []
            decay_parameter_list = []

            for n, p in loss_model.named_parameters():
                if not p.requires_grad:
                    continue
                if any(key in n for key in embedding_keys):
                    embedding_parameter_list.append(p)
                elif p.ndim <= 1 or any(nd in n for nd in no_decay):
                        no_decay_parameter_list.append(p)
                else:
                    decay_parameter_list.append(p)

            optimizer_grouped_parameters = [
                {'params': embedding_parameter_list, 'weight_decay': weight_decay},
                {'params': no_decay_parameter_list, 'weight_decay': 0.0},
                {'params': decay_parameter_list, 'weight_decay': weight_decay},
            ]
            embedding_flags = [1, 0, 0]

            embeddings = loss_model.model.get_embeddings()
            embedding_grad_scale = embedding_weight_decay_scale = None
            self.logger.info(f'current_token_mask > 0: {embeddings.get_current_token_mask().gt(0).sum()}')
            self.logger.info(f'unique_token_mask > 0: {embeddings.get_unique_token_mask().gt(0).sum()}')

            special_token_ids = None if not self.exclude_special_tokens \
                        else [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.unk_token_id, self.tokenizer.pad_token_id]

            if self.regularize_type in ['grad_decay', 'grad']:
                embedding_grad_scale = embeddings.get_embedding_grad_scale(self.grad_scale_type, special_token_ids)
                self.logger.info(f'embedding_grad_scale > 0: {embedding_grad_scale.gt(0).sum()}')
            if self.regularize_type in ['grad_decay', 'decay', 'fit_grad_decay']:
                embedding_weight_decay_scale = embeddings.get_embedding_weight_decay_scale(self.weight_decay_scale_type, special_token_ids)
                self.logger.info(f'embedding_weight_decay_scale > 0: {embedding_weight_decay_scale.gt(0).sum()}')

            optimizer = AdamW(
                optimizer_grouped_parameters, 
                embedding_flags=embedding_flags,
                embedding_grad_scale=embedding_grad_scale,
                embedding_weight_decay_scale=embedding_weight_decay_scale,
                **optimizer_params
            )
            return optimizer

        return super()._get_optimizer(loss_model, optimizer_class, optimizer_params, weight_decay, no_decay, high_lr_keys, high_lr_factor)
