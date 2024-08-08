from .CLDataset import CLDataset, MemoryDataset
from .ConcatDataset import ConcatDataset, get_concat_dataset_and_loader
from .utils import prepare_token_mapping


from zeronlg.datasets import (
    PretrainDataset,
    CaptionDataset,
    CaptionDatasetForRetrieval,
    TranslateDataset,
)
