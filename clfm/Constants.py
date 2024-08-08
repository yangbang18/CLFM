TF_ADAPTER_CONFIG_FN = 'adapter_config.json'
EMB_ADAPTER_CONFIG_FN = 'embedding_adapter_config.json'
EMB_ADAPTER_CKPT_FN = 'pytorch_embedding_adapter.bin'
EMB_CKPT_FN = 'embeddings.bin'
TOKEN_MAPPING_FN = 'token_mapping.json'
MEMORY_DATASET_FN = 'memory_dataset.json'
ADAPTER_ENCODER_CONFIG_FN = 'mTE_config.json'
PROMPT_CKPT_FN = 'prompt.bin'
FISHER_CKPT_FN = 'fisher.bin'


from .models.CLIPModel import CLIP_MODELS
from .models.MCLIP import MCLIP_MODELS
