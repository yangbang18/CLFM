from typing import Optional, Tuple, Union

import torch

from .modeling_clip import (
    CLIP_START_DOCSTRING,
    CLIPTextModelWithProjection,
    CLIPPreTrainedModel,
    CLIP_TEXT_INPUTS_DOCSTRING,
    replace_return_docstrings,
    CLIPTextModelOutput,
    CLIPTextConfig,
)
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward
from adapters.context import AdapterSetup
from adapters.heads import ModelWithFlexibleHeadsAdaptersMixin
from adapters.model_mixin import EmbeddingAdaptersWrapperMixin
from .init import init


@add_start_docstrings(CLIP_START_DOCSTRING)
class CLIPAdapterTextModelWithProjection(EmbeddingAdaptersWrapperMixin, ModelWithFlexibleHeadsAdaptersMixin, CLIPPreTrainedModel):
    _tied_weights_keys = []  # needs to be empty
    config_class = CLIPTextConfig

    def __init__(self, config):
        super().__init__(config)
        self.clip = CLIPTextModelWithProjection(config)
        init(self.clip)
        self._init_head_modules()
        #self.heads = None
        self.post_init()

    @add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        prompt: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # Added by Yang B.
        output_adapter_gating_scores=False,
        output_adapter_fusion_attentions=False,
        **kwargs
    ) -> Union[Tuple, CLIPTextModelOutput]:
        
        outputs, context = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            prompt=prompt,
            output_adapter_gating_scores=output_adapter_gating_scores,
            output_adapter_fusion_attentions=output_adapter_fusion_attentions,
            adapter_input_parallelized=kwargs.pop("adapter_input_parallelized", False),
            output_context=True,
        )
        # required e.g. for prompt tuning in all models
        kwargs["context"] = context
        return outputs
