import os
import json
import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, Tuple

from transformers import T5EncoderModel, AutoModel, AutoTokenizer
from adapters import AdapterConfig, PrefixTuningConfig, LoRAConfig, ConfigUnion
from adapters import parse_composition, AutoAdapterModel

from zeronlg.utils import get_cache_folder, download_if_necessary
from clfm.models.clip.adapter_model import CLIPAdapterTextModelWithProjection
from clfm import Constants
from .EmbeddingWrapper import EmbeddingWrapper
from clfm.utils.op_tokenizer import CLIPTokenizerFast


NAME_TO_MODEL_CLASS = {
    'google/byt5-small': T5EncoderModel,
    'facebook/nllb-200-distilled-1.3B': AutoModel,
    'openai/clip-vit-base-patch32': CLIPAdapterTextModelWithProjection,
    'openai/clip-vit-base-patch16': CLIPAdapterTextModelWithProjection,
    'openai/clip-vit-large-patch14': CLIPAdapterTextModelWithProjection,
}
NAME_TO_TOKENIZER_CLASS = {
    'openai/clip-vit-base-patch32': CLIPTokenizerFast,
    'openai/clip-vit-base-patch16': CLIPTokenizerFast,
    'openai/clip-vit-large-patch14': CLIPTokenizerFast,
}


class ModelMixin(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        assert hasattr(self, 'model_prefix'), 'please define self.model_prefix'

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def __repr__(self):
        return "{}(\n{}\n)".format(
            self.__class__.__name__, 
            '\n'.join([f"\t{k}: {v}" for k, v in self.get_config_dict().items()])
        )

    def get_transformer(self):
        if self.model_prefix:
            assert hasattr(self, self.model_prefix)
            return getattr(self, self.model_prefix)
        return self

    def get_embeddings(self) -> nn.Module:
        raise NotImplementedError()
    
    @property
    def embeddings(self) -> Union[nn.Module, EmbeddingWrapper]:
        return self.get_embeddings()
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def get_word_embeddings(self) -> nn.Module:
        try:
            return self.embeddings.get_word_embeddings()
        except:
            transformer = self.get_transformer()
            return transformer.get_input_embeddings()
    
    def get_word_embedding_dimension(self) -> int:
        return self.get_word_embeddings().weight.size(1)

    def get_sentence_embedding_dimension(self) -> int:
        transformer = self.get_transformer()
        return transformer.config.hidden_size

    def load_model_and_tokenizer(self,
                    model_name_or_path: str = None,
                    base_model_name: str = "distilbert-base-multilingual-cased",
                    cache_folder: str = get_cache_folder(),
                    use_auth_token: Union[bool, str, None] = None,
                    load_model: bool = True,
                    load_tokenizer: bool = True,
                    key_files: List[str] = ['config.json'],
                    ) -> nn.Module:
        
        model_class = NAME_TO_MODEL_CLASS.get(base_model_name, AutoAdapterModel)
        tokenizer_class = NAME_TO_TOKENIZER_CLASS.get(base_model_name, AutoTokenizer)

        try:
            model_path = download_if_necessary(model_name_or_path, cache_folder, use_auth_token, key_files=key_files)
            assert all(os.path.exists(os.path.join(model_path, fn)) for fn in key_files)
        except:
            model_path = download_if_necessary(base_model_name, cache_folder, use_auth_token, key_files=key_files)
            assert all(os.path.exists(os.path.join(model_path, fn)) for fn in key_files)
        
        model = model_class.from_pretrained(model_path) if load_model else None
        tokenizer = tokenizer_class.from_pretrained(model_path) if load_tokenizer else None

        return model, tokenizer

    def load_weights_(self, weights_path: str = None) -> None:
        if weights_path:
            path = os.path.join(weights_path, 'text_projection.bin')
            if os.path.exists(path):
                print('- Load text projection from', path)
                state_dict = torch.load(path, 'cpu')
                self.text_projection.load_state_dict(state_dict, strict=True)
            
            path = os.path.join(weights_path, Constants.EMB_CKPT_FN)
            if os.path.exists(path):
                print('- Load embeddings from', path)
                state_dict = torch.load(path, 'cpu')
                #assert all(k in state_dict for k in ['embedding_projection.weight', 'embedding_projection.bias'])
                msg = self.embeddings.load_state_dict(state_dict, strict=False)
                missing_keys, unexpected_keys = msg
                assert len(unexpected_keys) == 0, msg
            
            path = os.path.join(weights_path, 'backbone.bin')
            if os.path.exists(path):
                print('- Load backbone from', path)
                state_dict = torch.load(path, 'cpu')
                msg = self.get_transformer().load_state_dict(state_dict, strict=False)
                missing_keys, unexpected_keys = msg
                assert len(unexpected_keys) == 0, msg
            
            path = os.path.join(weights_path, Constants.PROMPT_CKPT_FN)
            if os.path.exists(path):
                print('- Load prompt from', path)
                state_dict = torch.load(path, 'cpu')
                self.prompt.load_state_dict(state_dict, strict=True)


class ModelWithAdapterMixin(ModelMixin):
    def __init__(self) -> None:
        super().__init__()
        self.model_is_trainable = True
        self.train_adapters = []
        self.train_fusion = None
        self.fusion_mapping = {}
    
    def set_token_mapping(self, token_mapping: Dict[int, List[str]] = None):
        assert self.adaptable_embeddings
        if token_mapping is not None:
            self.embeddings.token_mapping = token_mapping

    def freeze_model(self, freeze=True):
        """Freeze/unfreeze all weights of the model."""
        assert self.adaptable_transformer
        for name, param in self.get_transformer().base_model.named_parameters():
            if 'embedding_adapters' in name:
                continue
            param.requires_grad = not freeze
        self.model_is_trainable = not freeze
    
    def get_all_adapter_names(self) -> List[str]:
        embeddings_adapter_names = set()
        transformer_adapter_names = set()
        if self.adaptable_embeddings:
            embeddings_adapter_names = set(self.embeddings.adapter_names)
        if self.adaptable_transformer:
            transformer_adapter_names = set(self.get_transformer().adapters_config.adapters.keys())
        
        if len(embeddings_adapter_names) and len(transformer_adapter_names):
            assert embeddings_adapter_names == transformer_adapter_names
        
        return list(embeddings_adapter_names) or list(transformer_adapter_names)
    
    def get_all_adapter_fusion_names(self) -> List[str]:
        if self.adaptable_transformer:
            return list(self.get_transformer().adapters_config.fusions.keys())
        return []

    def load_adapters_from_paths(self, adapter_paths: Union[str, List[str]]) -> None:
        if type(adapter_paths) is str:
            adapter_paths = [adapter_paths]

        embeddings = self.get_embeddings()
        transformer = self.get_transformer()

        for path in adapter_paths:
            has_valid_adapter = False
            
            if os.path.exists(os.path.join(path, Constants.TF_ADAPTER_CONFIG_FN)):
                transformer.load_adapter(path)
                has_valid_adapter = True
            
            if os.path.exists(os.path.join(path, Constants.EMB_ADAPTER_CONFIG_FN)):
                assert isinstance(embeddings, EmbeddingWrapper)
                embeddings.load_adapter(path)
                has_valid_adapter = True
            
            assert has_valid_adapter, f'{path} does not contain any adapter'
    
    def load_adapters_from_path_and_names(self, path: str, names: Union[str, List[str]]):
        if type(names) is str:
            names = [names]
        adapter_paths = [os.path.join(path, name) for name in names]
        self.load_adapters_from_paths(adapter_paths)

    def load_adapters(self, paths, names=None):
        if type(paths) is str and names is not None:
            self.load_adapters_from_path_and_names(paths, names)
        else:
            self.load_adapters_from_paths(paths)
    
    def setup_adapter(self, 
                    adapter_name, 
                    adapter_config: Union[AdapterConfig, PrefixTuningConfig, LoRAConfig, ConfigUnion, None] = None, 
                    emb_adapter_config: Union[LoRAConfig, str, int, None]=None, 
                    add: bool=True,
                    set_active: bool=True,
                    set_train: bool=True, 
                    overwrite_ok: bool=False,
                    ) -> None:

        if add and adapter_config is not None:
            assert self.adaptable_transformer
            self.get_transformer().add_adapter(
                adapter_name, config=adapter_config, overwrite_ok=overwrite_ok)

        if add and emb_adapter_config is not None:
            assert self.adaptable_embeddings
            self.embeddings.add_adapter(
                adapter_name, config=emb_adapter_config, overwrite_ok=overwrite_ok)

        if set_active:
            self.set_active_adapter(adapter_name)

        if set_train:
            self.set_train_adapter(adapter_name)

    def set_active_adapter(self, adapter_name: str) -> Tuple[bool]:
        flag1 = flag2 = False
        if self.adaptable_embeddings:
            if adapter_name in self.embeddings.adapter_names:
                flag1 = True
                self.embeddings.set_active_adapter(adapter_name)
        
        if self.adaptable_transformer:
            transformer = self.get_transformer()
            if adapter_name in transformer.adapters_config.adapters:
                flag2 = True
                self.get_transformer().set_active_adapters([adapter_name])
        
        return (flag1, flag2)
    
    def set_train_adapter(self, adapter_name: str):
        assert self.adaptable_embeddings or self.adaptable_transformer

        self.model_is_trainable = False
        self.train_adapters = [adapter_name]
        
        # this will freeze all parameters in the base model
        # (including embeddings & backbone) other than the specific adapter
        if self.adaptable_transformer:
            transformer = self.get_transformer()
            if adapter_name in transformer.adapters_config.adapters:
                self.get_transformer().train_adapter(adapter_name)

        # embeddings.train_adapter must be called after transformer.train_adapter
        # so that the active embedding adapter can be trained
        if self.adaptable_embeddings:
            self.embeddings.train_adapter(adapter_name)

    def save_all_adapters(self, output_path: str):
        if self.adaptable_embeddings:
            self.embeddings.save_all_adapters(output_path)
        
        if self.adaptable_transformer:
            self.get_transformer().save_all_adapters(output_path)
    
    def delete_adapter(self, adapter_name: str):
        if self.adaptable_embeddings:
            self.embeddings.delete_adapter(adapter_name)
        
        if self.adaptable_transformer:
            self.get_transformer().delete_adapter(adapter_name)
    
    def delete_all_adapters(self):
        if self.adaptable_embeddings:
            self.embeddings.delete_all_adapters()

        if self.adaptable_transformer:
            transformer = self.get_transformer()
            names = list(transformer.adapters_config.adapters.keys())
            for adapter_name in names:
                transformer.delete_adapter(adapter_name)
    
    def setup_adapter_fusion(self, 
                             adapter_names: List[str],
                             fusion_type: str = 'static',
                             train_adapter_name: str = None,
                             identity: str = None,
                             ):
        if self.adaptable_transformer:
            transformer = self.get_transformer()
            # Add a fusion layer and tell the model to train fusion
            transformer.add_adapter_fusion(adapter_names, fusion_type)
            # By default, `train_adapter_fusion` only activate adapters and does not train them
            transformer.train_adapter_fusion([adapter_names])
            
            if train_adapter_name is not None:
                self.train_adapter_without_affecting_others(train_adapter_name)
                self.train_adatpers = [train_adapter_name]
            else:
                self.train_adapters = []
            
            # record info for saving and re-loading
            self.train_fusion = ','.join(adapter_names)
            assert self.train_fusion in self.get_all_adapter_fusion_names()

            if identity is not None:
                self.fusion_mapping[identity] = self.train_fusion

    def keep_current_adapter_fusion_only(self):
        if self.train_fusion is not None:
            for name in self.get_all_adapter_fusion_names():
                if name != self.train_fusion:
                    self.get_transformer().delete_adapter_fusion(name)

    def save_all_adapter_fusions(self, output_path: str):
        if self.adaptable_transformer:
            self.get_transformer().save_all_adapter_fusions(output_path)
    
    def load_adapter_fusion(self, path: str, set_active: bool=True):
        if self.adaptable_transformer:
            self.get_transformer().load_adapter_fusion(path, set_active=set_active)
    
    def load_adapter_fusion_wisely(self, 
                                   identity: str = None, 
                                   adapter_fusion_name: str = None, 
                                   set_active: bool = True, 
                                   weights_path: str = None) -> bool:
        
        weights_path = weights_path or getattr(self, 'weights_path', None)
        if weights_path is None:
            return False

        if identity is None and adapter_fusion_name is None:
            config = json.load(open(os.path.join(weights_path, Constants.ADAPTER_ENCODER_CONFIG_FN)))
            fusion_mapping = config.get('fusion_mapping', {})
            if not len(fusion_mapping):
                return False
            for k, v in fusion_mapping.items():
                self.load_adapter_fusion(os.path.join(weights_path, v), set_active)
                self.fusion_mapping[k] = v
            return fusion_mapping
        
        if identity is not None:
            if identity not in self.fusion_mapping:
                return False
            adapter_fusion_name = self.fusion_mapping[identity]
        elif adapter_fusion_name not in self.get_all_adapter_fusion_names():
            return False
    
        self.load_adapter_fusion(os.path.join(weights_path, adapter_fusion_name), set_active)
        return adapter_fusion_name

    def train_adapter_without_affecting_others(self, adapter_name: str):
        adapter_setup = parse_composition(adapter_name)
        transformer = self.get_transformer()

        adapter_setup = parse_composition(adapter_name)
        transformer.apply_to_adapter_layers(lambda i, layer: layer.enable_adapters(adapter_setup, True, False))
        for adapter_name in adapter_setup:
            if adapter_name in transformer.base_model.shared_parameters:
                for param in transformer.base_model.shared_parameters[adapter_name].values():
                    param.requires_grad = True

    @property
    def adaptable_embeddings(self) -> bool:
        return isinstance(self.embeddings, EmbeddingWrapper)
    
    @property
    def adaptable_transformer(self) -> bool:
        return True

    def summary(self, as_dict=False) -> Union[List[Dict[str, Any]], str]:
        # get the summary of adapters and the full model
        rows = self.model.adapter_summary(as_dict=True)

        full_model_item = rows[-1]
        full_model_item['train'] = self.model_is_trainable
        
        adapter_fusion_params = 0
        adapter_fusion_train = True
        for n, p in self.named_parameters():
            if 'adapter_fusion_layer' in n:
                adapter_fusion_params += p.numel()
                adapter_fusion_train &= p.requires_grad
        if adapter_fusion_params:
            full_model_item['#param'] -= adapter_fusion_params

        embeddings = self.get_embeddings()
        if isinstance(embeddings, EmbeddingWrapper):
            full_model_item['#param'] -= embeddings.adapter_params
            for row in rows:
                row['%param'] = row['#param'] * 100 / full_model_item['#param']
            
            emb_rows = embeddings.summary(as_dict=True, total_params=full_model_item['#param'])
            prepend = []
            for row in emb_rows:
                if row['name'].startswith('Embeddings'):
                    rows.insert(-1, row)
                else:
                    prepend.append(row)
            rows = prepend + rows
            embeddings_params = embeddings.total_params - embeddings.adapter_params
        else:
            embeddings = self.get_embeddings()
            word_embs = self.get_word_embeddings()
            embeddings_params = sum(p.numel() for p in embeddings.parameters())
            word_embs_params = sum(p.numel() for p in word_embs.parameters())

            rows.insert(-1, {
                'name': 'Embeddings-Word Embeds',
                '#param': word_embs_params,
                '%param': word_embs_params * 100 / full_model_item['#param'],
                'train': word_embs.weight.requires_grad,
            })
            rest_params = embeddings_params - word_embs_params

            if hasattr(embeddings, 'embedding_projection'):
                projection_params = sum(p.numel() for p in embeddings.embedding_projection.parameters())
                rows.insert(-1, {
                    'name': 'Embeddings-Projection',
                    '#param': projection_params,
                    '%param': projection_params * 100 / full_model_item['#param'],
                    'train': embeddings.embedding_projection.weight.requires_grad,
                })
                rest_params -= projection_params
            
            if rest_params > 0:
                is_trainable = False
                for n, p in embeddings.named_parameters():
                    if n.startswith('embedding_projection') or n.startswith('token_embedding') or n.startswith('word_embedding'):
                        continue
                    is_trainable = is_trainable or p.requires_grad

                rows.insert(-1, {
                    'name': 'Embeddings-Rest',
                    '#param': rest_params,
                    '%param': rest_params * 100 / full_model_item['#param'],
                    'train': is_trainable,
                })

        # adpter fusion
        if adapter_fusion_params:
            # TODO: train, active
            rows = [{
                'name': 'AdapterFusion',
                '#param': adapter_fusion_params,
                '%param': adapter_fusion_params * 100 / full_model_item['#param'],
                'train': adapter_fusion_train,
            }] + rows

        # backbone
        head_params = sum(p.numel() for p in self.text_projection.parameters())
        backbone_params = full_model_item['#param'] - embeddings_params
        if self.is_clip:
            backbone_params -= head_params
        rows.insert(-1, {
            'name': 'Backbone',
            '#param': backbone_params,
            '%param': backbone_params * 100 / full_model_item['#param'],
            'train': full_model_item['train'],
        })
        
        # head
        line_about_head = {
            'name': 'Head',
            '#param': head_params,
            '%param': head_params * 100 / full_model_item['#param'],
            'train': next(iter(self.text_projection.parameters())).requires_grad,
        }
        if self.is_clip:
            rows.insert(-1, line_about_head)
            # prompt pool
            if self.with_prompt:
                params = sum(p.numel() for p in self.prompt.parameters())
                rows.insert(0, {
                    'name': 'Prompt Pool',
                    '#param': params,
                    '%param': params * 100 / full_model_item['#param'],
                    'train': next(self.prompt.parameters()).requires_grad,
                })
        else:
            rows.append(line_about_head)

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
            sep_flag = True
            for i, row in enumerate(rows):
                if row['name'].startswith('Embeddings'):
                    if i == 0:
                        sep_flag = False
                    elif i != 0 and sep_flag:
                        sep_flag = False
                        s.append("-" * total_length)
                s.append(row_format.format(*[row.get(h, "") for h in header]))
            s.insert(len(s) - 1, "-" * total_length)
            s.append("=" * total_length)
            return "\n".join(s)

