from .CLIPModel import CLIPModel
from .MCLIP import MCLIP
from .AdapterEncoder import AdapterEncoder

from zeronlg.models import (
    Dense,
    Projector,
    Decoder,
    Transformer,
)

from sentence_transformers.models import Pooling

SBERT_MAPPINGS = {
    'sentence_transformers.models.CLIPModel': 'clfm.models.CLIPModel',
    'zeronlg.models.CLIPModel': 'clfm.models.CLIPModel',
}
