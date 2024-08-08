import torch
import json
import os
import logging
from torch import nn
from adapters import LoRAConfig
from adapters.methods.lora import LoRA
from typing import Union, List, Dict, Any
from collections import OrderedDict, defaultdict
from ... import Constants
            

class EmbeddingWrapper(nn.Module):
    def __init__(self,
                 clip_embeddings: nn.Module, 
                 pretrained_word_embeddings: nn.Module = None,
                 num_tokens: int = None,
                 **kwargs,
                 ) -> None:
        super().__init__()
        self.clip_vocab_size = clip_embeddings.token_embedding.weight.size(0)

        self.position_embeddings = clip_embeddings.position_embedding
        self.register_buffer("position_ids", clip_embeddings.position_ids)

        if pretrained_word_embeddings is not None:
            self.word_embeddings = pretrained_word_embeddings
        else:
            assert num_tokens is not None
            # in this case, we will load pre-trained weights after initialization
            self.word_embeddings = nn.Embedding(num_tokens, self.clip_embed_dim)

        if self.word_embed_dim != self.clip_embed_dim:
            self.embedding_projection = nn.Linear(self.word_embed_dim, self.clip_embed_dim)
            self.embedding_projection.weight.data.normal_(mean=0.0, std=0.02)
            self.embedding_projection.bias.data.zero_()
        else:
            self.embedding_projection = None

        # for regularizing gradients of word embeddings
        self.register_buffer("token_frequencies", torch.ones(self.vocab_size))
        self.token_mapping = defaultdict(list)

        self.old_vocab_size = self.word_embeddings.weight.size(0)
        self.reset()
    
    @property
    def vocab_size(self):
        return self.word_embeddings.weight.size(0)

    @property
    def word_embed_dim(self):
        return self.word_embeddings.weight.size(1)

    @property
    def token_embeddings(self):
        return self.word_embeddings

    @property
    def clip_embed_dim(self):
        return self.position_embeddings.weight.size(1)

    def get_word_embeddings(self):
        return self.word_embeddings

    def set_word_embeddings(self, word_embeddings: nn.Embedding):
        self.word_embeddings = word_embeddings

    def update_token_frequencies(self, 
                                 new_vocab_size: int, 
                                 old_vocab_size: int = None,
                                 plus_one_before_the_frist_task: bool = True,
                                 ):
        old_token_frequencies = self.token_frequencies
        
        if old_vocab_size is None:
            old_vocab_size = len(old_token_frequencies)
        else:
            assert len(old_token_frequencies) == old_vocab_size

        if old_vocab_size == self.clip_vocab_size and plus_one_before_the_frist_task:
            # this is the first time we expand clip's embedding
            # we add 1 to each element in `old_token_frequencies`, 
            # so that self.get_unique_token_mask() will not consider original tokens
            print('add 1 to each element in token_frequencies before expanding it')
            old_token_frequencies += 1

        new_token_frequencies = old_token_frequencies.new_ones(new_vocab_size)
        if new_vocab_size >= old_vocab_size: # dynamically expand vocabulary
            new_token_frequencies[:old_vocab_size] = old_token_frequencies
        else: # use language-specific vocabulary, in this case, new_token_frequencies takes no effects
            pass
        self.register_buffer("token_frequencies", new_token_frequencies)
    
    def forward_input_ids(self, input_ids):
        if self._active_adapter:
            lora: LoRA = self.embedding_adapters[self._active_adapter]
            delta_w = lora.lora_B @ lora.lora_A
            weight = lora.com(self.word_embeddings.weight, delta_w)
        else:
            if len(self.embedding_adapters):
                logging.warning("There are embedding adapters available but none are activated for the forward pass.")

            weight = self.word_embeddings.weight

        # inputs_embeds = self.word_embeddings(input_ids)
        inputs_embeds = torch.embedding(weight, input_ids)
        if self.embedding_projection is not None:
            inputs_embeds = self.embedding_projection(inputs_embeds)
        
        return inputs_embeds
    
    def forward(self, input_ids, position_ids=None, **kwargs):
        inputs_embeds = self.forward_input_ids(input_ids)

        if position_ids is None:
            seq_length = input_ids.shape[-1]
            position_ids = self.position_ids[:, :seq_length]
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings

        return embeddings

    def reset(self):
        self._active_adapter = None
        self.embedding_adapters = nn.ModuleDict()
    
    def add_adapter(self, 
            name: str, 
            config: Union[LoRAConfig, str, int], 
            overwrite_ok: bool=False, 
            ) -> None:

        if name in self.embedding_adapters and not overwrite_ok:
            raise ValueError(f'`{name}` has been existed in embedding_adapters but `overwrite_ok` is False')
        
        if type(config) is int:
            r = config
            config = LoRAConfig.load({"architecture": "lora", "r": r, "alpha": r})
        elif type(config) is str:
            config = LoRAConfig.load(config)
        
        lora_A_shape = (config.r, self.word_embed_dim)
        lora_B_shape = (self.vocab_size, config.r)
        lora = LoRA(lora_A_shape, lora_B_shape, config)
        lora.config = config

        self.embedding_adapters[name] = lora
    
    def delete_adapter(self, name: str):
        if name in self.embedding_adapters:
            if self.name == self._active_adapter:
                self._active_adapter = None
            self.embedding_adapters.pop(name)
    
    def delete_all_adapters(self):
        self.reset()
    
    def train_adapter(self, name: str):
        # Different from train_adapter in adapter-transformers
        # this function only affects the training status of self.embedding_adapters
        for _ in self.embedding_adapters:
            for param in self.embedding_adapters[_].parameters():
                param.requires_grad = False if _ != name else True
    
    def set_active_adapter(self, name: str=None):
        if not name:
            self._active_adapter = None
        elif name in self.embedding_adapters:
            self._active_adapter = name

    def save_adapter(self, name: str, save_path: str):
        lora = self.embedding_adapters[name]
        os.makedirs(save_path, exist_ok=True)

        EmbeddingLoRAKeys = ["architecture", "r", "alpha", "dropout", "composition_mode", "init_weights"]

        with open(os.path.join(save_path, Constants.EMB_ADAPTER_CONFIG_FN), 'w') as wf:
            # we only save meaningful configurations
            config = lora.config.to_dict()
            config = {k: v for k, v in config.items() if k in EmbeddingLoRAKeys}
            config['name'] = name
            json.dump(config, wf)
        
        torch.save(
            lora.state_dict(),
            os.path.join(save_path, Constants.EMB_ADAPTER_CKPT_FN)
        )
    
    def save_all_adapters(self, path: str):
        for name in self.embedding_adapters:
            save_path = os.path.join(path, name)
            self.save_adapter(name, save_path)
    
    def load_adapter(self, path: str):
        config_path = os.path.join(path, Constants.EMB_ADAPTER_CONFIG_FN)
        ckpt_path = os.path.join(path, Constants.EMB_ADAPTER_CKPT_FN)
        assert os.path.exists(config_path)
        assert os.path.exists(ckpt_path)
        
        config = json.load(open(config_path))
        name = config.pop('name')
        config = LoRAConfig.load(config)
        
        self.add_adapter(name, config, overwrite_ok=True)
        
        state_dict = torch.load(ckpt_path, 'cpu')
        self.embedding_adapters[name].load_state_dict(state_dict)
    
    def has_adapters(self) -> bool:
        return len(self.embedding_adapters) > 0
    
    def get_current_token_mask(self) -> torch.FloatTensor:
        if not hasattr(self, 'current_token_mask'):
            # record tokens appeared in the current task
            ids = [int(tokenID) for tokenID in self.token_mapping]
            self.current_token_mask = torch.zeros(self.vocab_size, device=self.device)
            self.current_token_mask[ids] = 1 
        return self.current_token_mask # (vocab_size,)

    def get_unique_token_mask(self, frequency_subtract_current: bool = False) -> torch.FloatTensor:
        if not hasattr(self, 'unique_token_mask'):
            # record unique tokens appeared in the current task
            token_frequencies = self.token_frequencies.to(self.device)
            current_token_mask = self.get_current_token_mask()
            if frequency_subtract_current:
                token_frequencies = token_frequencies - current_token_mask
            self.unique_token_mask = token_frequencies.eq(1) & current_token_mask.eq(1)
            self.unique_token_mask = self.unique_token_mask.float()
        return self.unique_token_mask # (vocab_size,)

    def _get_scale(self, scale_type: str="current_reciprocal", special_token_ids: List[int] = None) -> torch.Tensor:
        token_frequencies = self.token_frequencies.to(self.device).unsqueeze(1)
        current_token_mask = self.get_current_token_mask().unsqueeze(1)
        unique_token_mask = self.get_unique_token_mask().unsqueeze(1)

        if scale_type == 'current':
            scale = current_token_mask
        elif scale_type == 'current_reciprocal':
            scale = current_token_mask / token_frequencies
        elif scale_type == 'unique':
            scale = unique_token_mask
        elif scale_type == 'current_reciprocal_rescale':
            scale = current_token_mask / token_frequencies
            mask = scale.eq(1.0) # unique tokens
            now_scale_sum = scale.sum()
            org_scale_sum = current_token_mask.sum()
            coeff = org_scale_sum / now_scale_sum
            scale[mask] = scale[mask] * coeff
        elif scale_type == 'current_reciprocal_rescale_log2':
            scale = current_token_mask / token_frequencies
            mask = scale.eq(1.0) # unique tokens
            now_scale_sum = scale.sum()
            org_scale_sum = current_token_mask.sum()
            coeff = org_scale_sum / now_scale_sum
            if coeff < 2:
                coeff = 1.0
            else:
                coeff = torch.log2(coeff)
            scale[mask] = scale[mask] * coeff
        else:
            assert "fixed_" in scale_type or "addend_" in scale_type, f"{scale_type}"
            denominator = int(scale_type.split('_')[1])
            assert denominator > 0
            mask = token_frequencies.gt(1)
            scale = 1 / token_frequencies
            if "fixed" in scale_type:
                scale[mask] = 1 / denominator
            else:
                scale[mask] = 1 / (token_frequencies[mask] + denominator)
            scale.mul_(current_token_mask)
        
        if special_token_ids is not None:
            for token_id in special_token_ids:
                scale[token_id] = 0
        
        return scale
    
    def get_embedding_grad_scale(self, grad_scale_type: str="current_reciprocal", special_token_ids: List[int] = None) -> torch.Tensor:
        if not hasattr(self, 'embedding_grad_scale'):
            self.embedding_grad_scale = self._get_scale(grad_scale_type, special_token_ids)
        return self.embedding_grad_scale

    def get_embedding_weight_decay_scale(self, weight_decay_scale_type: str="current", special_token_ids: List[int] = None) -> torch.Tensor:
        if not hasattr(self, 'embedding_weight_decay_scale'):
            self.embedding_weight_decay_scale = self._get_scale(weight_decay_scale_type, special_token_ids)
        return self.embedding_weight_decay_scale

    @property
    def adapter_names(self) -> List[str]:
        return list(self.embedding_adapters.keys())

    @property
    def adapter_params(self):
        return sum(p.numel() for p in self.embedding_adapters.parameters())
    
    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def device(self):
        return next(self.parameters()).device
    
    def summary(self, as_dict=False, total_params=None)-> Union[List[Dict[str, Any]], str]:
        rows = []

        embeddings_params = sum(p.numel() for p in self.parameters())
        total_params = total_params or (embeddings_params - self.adapter_params)

        for name in self.embedding_adapters:
            params = sum(p.numel() for p in self.embedding_adapters[name].parameters())
            rows.append({
                'name': f'{name} (Embedding)',
                'architecture': 'lora',
                '#param': params,
                '%param': params * 100 / total_params,
                'train': name == self._active_adapter,
                'active': name == self._active_adapter,
            })

        for name, module in zip(
            ['Embeddings-Word Embeds', 'Embeddings-Projection', 'Embeddings-Pos Embeds'],
            [self.word_embeddings, self.embedding_projection, self.position_embeddings],
        ):
            if module is None or isinstance(module, nn.Identity):
                continue
            
            params = sum(p.numel() for p in module.parameters())

            rows.append({
                'name': name,
                '#param': params,
                '%param': params * 100 / total_params,
                'train': next(module.parameters()).requires_grad,
            })
        
        if as_dict:
            return rows
        else:
            # print
            header = ["name", "architecture", "#param", "%param", "active", "train"]
            total_length = 80
            header_format = "{:<25}{:<15}{:>12}{:>12}{:>8}{:>8}"
            row_format = "{:<25}{:<15}{:>12,}{:>12.3f}{:>8}{:>8}"
            s = ["=" * total_length]
            s.append(header_format.format(*map(lambda x: x.title(), header)))
            s.append("-" * total_length)
            for row in rows:
                s.append(row_format.format(*[row.get(h, "") for h in header]))
            s.append("=" * total_length)
            return "\n".join(s)

    def save(self, path: bool, shrink_embeddings: bool = False, skip_adapters: bool = True, only_return_save_obj: bool = False):
        save_obj = OrderedDict()

        for k, v in self.named_parameters():
            if skip_adapters and 'embedding_adapters' in k:
                continue
            save_obj[k] = v
        
        # update token frequencies for saving
        token_frequencies_copy = self.token_frequencies.clone()
        for tokenID, langauges in self.token_mapping.items():
            self.token_frequencies[int(tokenID)].add_(len(langauges))

        save_obj['token_frequencies'] = self.token_frequencies
        
        # restore token frequencies for training another epoch
        self.token_frequencies = token_frequencies_copy

        if shrink_embeddings \
            and self.word_embeddings.weight.requires_grad \
            and 'embedding_projection.weight' in save_obj:

            word_embeddings_weight = save_obj.pop('word_embeddings.weight')
            embedding_projection_weight = save_obj.pop('embedding_projection.weight')
            embedding_projection_bias = save_obj.pop('embedding_projection.bias')
            word_embeddings_weight = \
                word_embeddings_weight @ embedding_projection_weight.t() + embedding_projection_bias

            save_obj['word_embeddings.weight'] = word_embeddings_weight

        if only_return_save_obj:
            return save_obj        
        
        torch.save(save_obj, os.path.join(path, Constants.EMB_CKPT_FN))
        
        if len(self.token_mapping):
            with open(os.path.join(path, Constants.TOKEN_MAPPING_FN), 'w') as wf:
                json.dump(self.token_mapping, wf)

    @staticmethod
    def load(
            clip_embeddings: nn.Module,
            pretrained_word_embeddings: nn.Module,
            path: str=None, 
            num_tokens: int = None,
            ):
        
        module = EmbeddingWrapper(
            clip_embeddings, 
            pretrained_word_embeddings, 
            num_tokens=num_tokens,
        )
        if path is None:
            return module

        ########## Load Embedding Weights
        emb_file = os.path.join(path, Constants.EMB_CKPT_FN)
        print('- Load embeddings from', emb_file)
        state_dict = torch.load(emb_file, 'cpu')
        if 'embedding_projection.weight' not in state_dict \
            or state_dict['word_embeddings.weight'].shape != pretrained_word_embeddings.weight.shape:
            if module.embedding_projection is None or isinstance(module.embedding_projection, nn.Identity):
                pass
            else:
                module.word_embeddings = nn.Embedding(*state_dict['word_embeddings.weight'].shape)
                module.embedding_projection = nn.Identity()
        
        msg = module.load_state_dict(state_dict, strict=False)
        missing_keys, unexpected_keys = msg
        assert len(unexpected_keys) == 0, msg

        ########## Load Token Mapping
        mapping_file = os.path.join(path, Constants.TOKEN_MAPPING_FN)
        if os.path.exists(mapping_file):
            module.token_mapping = json.load(open(mapping_file))

        return module
