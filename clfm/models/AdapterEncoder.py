import os
import json
import torch
from torch import nn
from typing import Union, List, Tuple
from sentence_transformers.models import Pooling
from collections import OrderedDict
from transformers import AutoTokenizer
from zeronlg.models import Dense
from ..utils import get_cache_folder
from ..Constants import *
from .continual_learning import EmbeddingWrapper,ModelWithAdapterMixin


class AdapterEncoder(ModelWithAdapterMixin):
    """AdapterEncoder

    :param model_name_or_path: Huggingface's model name or local path that stores (newly) pre-trained transformer
    :param base_model_name: Huggingface's model name, the initial weights of transformer before continual learning
    :param embedding_name: Huggingface's model name that provides embeddings, concated with base_model_name's backbone
    :param weights_path: local path that stores weights of the embedding layer and/or the projection layer
    """

    model_prefix: str = 'model'
    def __init__(self,  
                 model_name_or_path: str = None,
                 base_model_name: str = "distilbert-base-multilingual-cased",
                 embedding_name: str = None, 
                 weights_path: str = None,
                 max_seq_length: int = 128,
                 clip_model_name: str = None,
                 dim_latent_space: int = 512,
                 cache_folder: str = get_cache_folder(),
                 use_auth_token: Union[bool, str, None] = None,
                 shrink_embeddings: bool = True,
                 dim_word: int = None,
                 mean_std_word: Tuple[float, float] = None,
                 dynamic_vocab: bool = False,
                 with_prompt: bool = False,
                 prompt_type: str = 'CodaPrompt',
                 prompt_config: str = 'default',
                 **kwargs,
                 ):
        super(AdapterEncoder, self).__init__()
        self.config_keys = [
            'model_name_or_path', 'base_model_name', 'weights_path', 'max_seq_length', 'clip_model_name', 'dim_latent_space', 'cache_folder', 'dynamic_vocab',
        ]
        self.model_name_or_path = model_name_or_path
        self.base_model_name = base_model_name
        self.weights_path = weights_path
        self.max_seq_length = max_seq_length
        self.clip_model_name = clip_model_name
        self.dim_latent_space = dim_latent_space
        self.cache_folder = cache_folder
        self.dynamic_vocab = dynamic_vocab
        self.shrink_embeddings = shrink_embeddings

        self.model, self.tokenizer = self.load_model_and_tokenizer(
            model_name_or_path, base_model_name, cache_folder, use_auth_token)

        self.is_clip = self.base_model_name in CLIP_MODELS
        if self.is_clip:
            self.embedding_name = embedding_name
            self.max_seq_length = 77
            self.dim_word = dim_word
            self.with_prompt = with_prompt
            self.prompt_type = prompt_type
            self.prompt_config = prompt_config

            self.config_keys.insert(2, 'embedding_name')
            self.config_keys.append('shrink_embeddings')
            if self.dim_word:
                self.config_keys.append('dim_word')
                self.config_keys.append('mean_std_word')
            if self.with_prompt:
                self.config_keys.append('with_prompt')
                self.config_keys.append('prompt_type')
                self.config_keys.append('prompt_config')

                from clfm.adapters import DualPrompt, CodaPrompt
                PROMPT_CLASS = eval(prompt_type)

                self.prompt = PROMPT_CLASS(
                    prompt_config=prompt_config,
                    embed_dim=self.model.base_model.text_model.config.hidden_size,
                )
            
            assert base_model_name == clip_model_name, f'base_model `{base_model_name}` should be the same as clip_model `{clip_model_name}`'
            assert self.model.config.projection_dim == dim_latent_space, \
                    f'hidden size of the student ({self.model.config.hidden_size}) does not equal to that of the teacher ({dim_latent_space})'

            if not model_name_or_path:
                # we have not saved the CLIP model with the modified embedding yet
                # so we should replace CLIP's English embeddings with a multilingual one
                assert embedding_name

                if self.dynamic_vocab:
                    assert weights_path is not None, \
                        "Please do not set `dynamic_vocab` to True by yourself. Instead, you should call `update_tokenizer`."
                    self.tokenizer = AutoTokenizer.from_pretrained(weights_path)
                    pretrained_word_embeddings = None
                else:
                    lang_model, self.tokenizer = self.load_model_and_tokenizer(
                        embedding_name, embedding_name, cache_folder, use_auth_token)

                    if mean_std_word is not None:
                        if -1 in mean_std_word:
                            word_embeddings = self.get_word_embeddings()
                            mean, std = word_embeddings.weight.mean().item(), word_embeddings.weight.std().item()
                        else:
                            mean, std = mean_std_word
                    else:
                        mean, std = 0, 0.02

                    if self.dim_word:
                        self.mean_std_word = (mean, std)
                        print('randomly initialize pretrained_word_embeddings')
                        pretrained_word_embeddings = nn.Embedding(
                            self.tokenizer.vocab_size, self.dim_word, padding_idx=self.tokenizer.pad_token_id)
                        
                        pretrained_word_embeddings.weight.data.normal_(mean=mean, std=std)
                    else:
                        pretrained_word_embeddings = lang_model.get_input_embeddings()
                        if mean_std_word is not None and -1 in mean_std_word:
                            print('recalibrate mean & std of pretrained_word_embeddings')
                            inputs_embeds = pretrained_word_embeddings.weight.data
                            now_mean, now_std = inputs_embeds.mean(), inputs_embeds.std()
                            pretrained_word_embeddings.weight.data = (inputs_embeds - (now_mean - mean)) * std / now_std
                            inputs_embeds = pretrained_word_embeddings.weight.data
                            print(inputs_embeds.mean(), inputs_embeds.std())

                self.model.base_model.text_model.embeddings = EmbeddingWrapper.load(
                    clip_embeddings=self.get_embeddings(),
                    pretrained_word_embeddings=pretrained_word_embeddings,
                    path=weights_path,
                    num_tokens=len(self.tokenizer),
                )
                self.model.config.vocab_size = self.tokenizer.vocab_size
                
            self.text_projection = self.model.base_model.text_projection
        else:
            self.pooling = Pooling(self.model.config.hidden_size)
            self.text_projection = Dense(self.model.config.hidden_size, dim_latent_space, activation_function=nn.modules.linear.Identity())
        
        self.load_weights_(weights_path)

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id or self.tokenizer.sep_token_id

    def tokenize(self, texts: List[str], **kwargs):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_seq_length)
        if self.dynamic_vocab:
            assert self.is_clip
            # When we dynamically expand the vocab, the eos_token_id is no longer the largest index in the vocab
            eos_ids = inputs['input_ids'].new_zeros(inputs['input_ids'].size(0))
            for i, ids in enumerate(inputs['input_ids']):
                eos_ids[i] = ids.tolist().index(self.eos_token_id)
            inputs['eos_ids'] = eos_ids
        return inputs
    
    def forward(self, features):
        inputs = dict(
            input_ids=features.get('input_ids'),
            attention_mask=features.get('attention_mask', None),
            # position_ids=features.get('position_ids', None),
            output_attentions=features.get('output_attentions', None),
            output_hidden_states=features.get('output_hidden_states', None),
            return_dict=False,
        )

        if self.is_clip:
            if self.is_clip and self.with_prompt and features.get('with_prompt', True):
                query = features.get('cls_features', None)
                assert query is not None
                prompt, loss = self.prompt(query)
                features['loss_prompt'] = loss
                inputs['prompt'] = prompt

            inputs['output_attentions'] = True
            inputs['return_dict'] = True
            outputs = self.model(**inputs)
            text_embeds = outputs.text_embeds
            last_hidden_state = outputs.last_hidden_state

            B = last_hidden_state.shape[0]
            if self.base_model_name == self.embedding_name:
                # CLIP model itself
                if not self.dynamic_vocab:
                    # eos_token_id is exactly the maximun index
                    features['sentence_embedding'] = text_embeds
                else:
                    # eos_token_id is not the maximun index
                    eos_embeds = last_hidden_state[torch.arange(B), features['eos_ids']]
                    features['sentence_embedding'] = self.text_projection(eos_embeds)
            else:
                eos_embeds = last_hidden_state[torch.arange(B), inputs['input_ids'].eq(self.eos_token_id).nonzero()[:, 1]]
                features['sentence_embedding'] = self.text_projection(eos_embeds)
            
            features['token_embeddings'] = last_hidden_state
            features['attentions'] = outputs.attentions
        else:
            hidden_states, *_ = self.model(**inputs)
            features['token_embeddings'] = hidden_states
            features = self.pooling(features)
            features = self.text_projection(features)
        return features

    def get_embeddings(self) -> nn.Module:
        try:
            if self.is_clip:
                embeddings = self.model.base_model.text_model.embeddings
            else:
                embeddings = self.model.base_model.embeddings
        except:
            embeddings = None
        return embeddings

    def update_tokenizer(self, 
                         tokenizer_path: str, 
                         save_path: str, 
                         mean: float = 0, 
                         std: float = 0.02, 
                         msg_prefix: str = '',
                         directly_use_new_tokenizer: bool = False,
                         plus_one_before_the_frist_task: bool = True,
                         ) -> str:
        assert self.embedding_name in CLIP_MODELS
        assert self.embedding_name == self.base_model_name
        assert self.adaptable_embeddings
        
        from ..utils import combine_tokenizers
        
        msg = []

        data = combine_tokenizers(
            self.tokenizer, tokenizer_path, save_path, as_dict=True, 
            directly_use_new_tokenizer=directly_use_new_tokenizer,
        )
        new_tokenizer = data.pop('tokenizer')
        for k, v in data.items():
            msg.append(f'{msg_prefix}{k}: {v}')
        
        old_vocab_size = len(self.tokenizer)
        new_vocab_size = len(new_tokenizer)
        msg.append(f'{msg_prefix}old_vocab_size: {old_vocab_size}; new_vocab_size: {new_vocab_size}')

        self.embeddings.update_token_frequencies(
            new_vocab_size=new_vocab_size, 
            old_vocab_size=old_vocab_size,
            plus_one_before_the_frist_task=plus_one_before_the_frist_task,
        )

        old_embeddings = self.get_word_embeddings()
        new_embeddings = nn.Embedding(new_vocab_size, old_embeddings.weight.size(1))
        if mean == -1 and std == -1:
            mean = old_embeddings.weight.mean().item()
            std = old_embeddings.weight.std().item()
            msg.append(f'{msg_prefix}initialize new_embeddings with (mean, std) of the old_embeddings')
        new_embeddings.weight.data.normal_(mean=mean, std=std)

        if directly_use_new_tokenizer:
            # copy embeddings in the intersection
            old_vocab = self.tokenizer.get_vocab()
            new_vocab = new_tokenizer.get_vocab()
            overlapped_tokens = set(old_vocab.keys()) & set(new_vocab.keys())
            for token in overlapped_tokens:
                old_idx = old_vocab[token]
                new_idx = new_vocab[token]
                new_embeddings.weight.data[new_idx] = old_embeddings.weight.data[old_idx]
            msg.append(f'{msg_prefix}num_overlap: {len(overlapped_tokens)}')
            msg.append(f'{msg_prefix}overlap_ratio: {len(overlapped_tokens) * 100 / new_vocab_size}')
        else:
            assert new_vocab_size >= old_vocab_size
            new_embeddings.weight.data[:old_vocab_size, :] = old_embeddings.weight.data[:old_vocab_size, :]

        self.embeddings.set_word_embeddings(new_embeddings)
        msg.append(f'{msg_prefix}mean: {mean}; std: {std}')

        self.tokenizer = new_tokenizer
        self.model.config.vocab_size = new_vocab_size
        self.model.vocab_size = new_vocab_size
        self.dynamic_vocab = True

        return '\n'.join(msg)
    
    def save_trainable_embeddings(self, output_path: str):
        if self.adaptable_embeddings:
            self.embeddings.save(
                path=output_path,
                shrink_embeddings=self.shrink_embeddings,
                skip_adapters=True
            )
    
    def save(self, output_path: str):
        self.save_all_adapters(output_path)
        self.save_all_adapter_fusions(output_path)

        if self.model_is_trainable:
            # self.delete_all_adapters()

            flag = self.is_clip and self.adaptable_embeddings
            if flag and not self.shrink_embeddings:
                embeddings_state_dict = self.embeddings.state_dict()
                state_dict = OrderedDict()
                for n, p in self.model.named_parameters():
                    if not any ([k in n for k in embeddings_state_dict]) and p.requires_grad:
                        state_dict[n] = p
                torch.save(state_dict, os.path.join(output_path, 'backbone.bin'))
                self.save_trainable_embeddings(output_path)
            else:
                state_dict = self.model.state_dict()
                if flag:
                    state_dict = {k: v for k, v in state_dict.items() if 'text_model.embeddings' not in k}
                    obj = self.embeddings.save(
                        path=output_path,
                        shrink_embeddings=self.shrink_embeddings,
                        skip_adapters=True,
                        only_return_save_obj=True,
                    )
                    state_dict['text_model.embeddings.token_embedding.weight'] = obj.pop('word_embeddings.weight')
                    state_dict['text_model.embeddings.position_embedding.weight'] = obj.pop('position_embeddings.weight')
                
                self.model.save_pretrained(output_path, state_dict=state_dict)
                self.tokenizer.save_pretrained(output_path)
        else:
            self.save_trainable_embeddings(output_path)

        if hasattr(self, 'text_projection') and next(iter(self.text_projection.parameters())).requires_grad:
            torch.save(self.text_projection.state_dict(), os.path.join(output_path, 'text_projection.bin'))

        with open(os.path.join(output_path, ADAPTER_ENCODER_CONFIG_FN), 'w') as fOut:
            config = self.get_config_dict()
            config['train_adapters'] = getattr(self, 'train_adapters', [])
            config['available_adapters'] = self.get_all_adapter_names()
            config['train_fusion'] = getattr(self, 'train_fusion', None)
            config['available_fusions'] = self.get_all_adapter_fusion_names()
            config['fusion_mapping'] = self.fusion_mapping
            json.dump(config, fOut, indent=2)

        if self.dynamic_vocab:
            self.tokenizer.save_pretrained(output_path)
        
        if self.is_clip and self.with_prompt:
            torch.save(self.prompt.state_dict(), os.path.join(output_path, PROMPT_CKPT_FN))

    @staticmethod
    def load(input_path: str):
        config_path = os.path.join(input_path, ADAPTER_ENCODER_CONFIG_FN)
        config = json.load(open(config_path))
        config['weights_path'] = input_path
        if 'cache_folder' not in config or not os.path.exists(config['cache_folder']):
            config['cache_folder'] = get_cache_folder()

        if os.path.exists(os.path.join(input_path, 'pytorch_model.bin')):
            config['model_name_or_path'] = input_path
        else:
            # `input_path` does not contain weights of pre-trained transformer
            pass

        model = AdapterEncoder(**config)

        # load all available adapters stored in the `input_path`
        available_adapters = config.get('available_adapters', [])
        if available_adapters:
            adapter_paths = [os.path.join(input_path, name) for name in available_adapters]
            model.load_adapters(adapter_paths)
        elif os.path.exists(os.path.join(input_path, TF_ADAPTER_CONFIG_FN)):
            model.load_adapters([input_path])

        # load the previously trained adapter and set it to be active during feed-forwarding
        train_adapters = config.get('train_adapters', []) or ([config['adapter_name']] if 'adapter_name' in config else [])
        assert len(train_adapters) in [0, 1]

        train_fusion = config.get('train_fusion', None)
        if train_fusion is not None:
            model.train_fusion = train_fusion
            model.fusion_mapping = config['fusion_mapping']
            available_fusions = config.get('available_fusions', [train_fusion])
            for fusion in available_fusions:
                model.load_adapter_fusion(os.path.join(input_path, train_fusion), set_active=fusion == train_fusion)
            if len(train_adapters) == 1:
                #model.train_adapter_without_affecting_others(train_adapters[0])
                train_adapters = []
        
        if len(train_adapters) == 1:
            model.setup_adapter(
                train_adapters[0], 
                adapter_config=None, 
                set_active=True, 
                set_train=True
            )

        return model
