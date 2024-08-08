import torch
from typing import Optional, Union, Callable
from zeronlg import ZeroNLG
from .Framework import Framework


class Seq2Seq(ZeroNLG):
    def __init__(self, 
                 multilingual_model: Union[str, Framework], 
                 clip_model: Union[str, Framework, None] = None, 
                 use_clip_tokens: Optional[bool] = None,
                 load_clip_model: bool = True,
                 device: Union[str, torch.device, None] = None,
                 framework_cls: Callable = Framework,
        ):
        super().__init__(multilingual_model, clip_model, use_clip_tokens, load_clip_model, device, framework_cls)
