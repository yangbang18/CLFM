import transformers
from typing import Callable
from torch import nn
from zeronlg.models import CLIPModel as BASE
from .clip.modeling_clip import CLIPModel as CLIP


CLIP_MODELS = [
    'openai/clip-vit-base-patch32',
    'openai/clip-vit-base-patch16',
    'openai/clip-vit-large-patch14',
]


class CLIPModel(BASE):
    def __init__(self, 
                 model_name: str = "openai/clip-vit-base-patch16", 
                 processor_name = None, 
                 use_clip_tokens: bool = False,
                 model_cls: Callable = CLIP,
                 processor_cls: Callable = transformers.CLIPProcessor,
    ):
        super().__init__(model_name, processor_name, use_clip_tokens, model_cls, processor_cls)

    def get_embeddings(self) -> nn.Module:
        return self.model.text_model.embeddings

    def get_input_embeddings(self) -> nn.Module:
        return self.model.text_model.embeddings.token_embedding
