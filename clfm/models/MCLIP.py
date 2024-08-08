import os
import torch
import transformers
from torch import nn
from typing import List
from multilingual_clip import Config_MCLIP


MCLIP_MODELS = [
    'M-CLIP/XLM-Roberta-Large-Vit-B-32',
    'M-CLIP/XLM-Roberta-Large-Vit-L-14',
    'M-CLIP/LABSE-Vit-L-14',
    'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus',
]

MCLIP2Vision = {
    'XLM-Roberta-Large-Vit-B-32': 'openai/clip-vit-base-patch32',
    'XLM-Roberta-Large-Vit-L-14': 'openai/clip-vit-large-patch14',
    'LABSE-Vit-L-14': 'openai/clip-vit-large-patch14',
    'XLM-Roberta-Large-Vit-B-16Plus': 'ViT-B-16-plus-240', # Not supported yet
}



class MultilingualCLIP(transformers.PreTrainedModel):
    config_class = Config_MCLIP.MCLIPConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        model_name_or_path = os.path.join(os.path.dirname(config._name_or_path), config.modelBase)
        if not os.path.exists(model_name_or_path):
            model_name_or_path = config.modelBase

        self.transformer = transformers.AutoModel.from_pretrained(model_name_or_path)
        self.LinearTransformation = torch.nn.Linear(in_features=config.transformerDimensions,
                                                    out_features=config.numDims)

    def forward(self, txt, tokenizer):
        txt_tok = tokenizer(txt, padding=True, return_tensors='pt')
        embs = self.transformer(**txt_tok)[0]
        att = txt_tok['attention_mask']
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        return self.LinearTransformation(embs)

    @classmethod
    def _load_state_dict_into_model(cls, model, state_dict, pretrained_model_name_or_path, _fast_init=True):
        model.load_state_dict(state_dict)
        return model, [], [], []



class MCLIP(nn.Module):
    def __init__(self,  model_name: str = "M-CLIP/XLM-Roberta-Large-Vit-B-32"):
        super(MCLIP, self).__init__()
        self.config_keys = []
        self.model = MultilingualCLIP.from_pretrained(model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        # for automatically loading the vision model in Seq2Seq
        self.clip_model_name = MCLIP2Vision[os.path.basename(model_name.rstrip('/').replace('_', '/'))]
        assert 'openai' in self.clip_model_name, 'only support loading CLIP models right now'

    def __repr__(self):
        return "MCLIP()"

    def forward(self, features):
        embs = self.model.transformer(
            input_ids=features['input_ids'],
            attention_mask=features['attention_mask'],
        )[0]
        att = features['attention_mask']
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        features['sentence_embedding'] = self.model.LinearTransformation(embs)
        return features

    def tokenize(self, texts: List[str], **kwargs):
        return self.tokenizer(texts, padding=True, return_tensors='pt')
    
    def get_embeddings(self) -> nn.Module:
        return self.model.transformer.embeddings

    def get_input_embeddings(self) -> nn.Module:
        return self.model.transformer.get_input_embeddings()
    
    def get_word_embedding_dimension(self) -> int:
        return self.model.transformer.config.hidden_size
    
    def get_sentence_embedding_dimension(self) -> int:
        return self.model.transformer.config.hidden_size
    
    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
    
    @staticmethod
    def load(input_path: str):
        return MCLIP(model_name=input_path)
