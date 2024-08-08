import os
import gzip
import json
import random
import logging
import numpy as np
from clfm.Constants import MEMORY_DATASET_FN
from torch import Tensor
from sentence_transformers import LoggingHandler
from torch.utils.data import Dataset
from typing import Union, List, Tuple


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

global_logger = logging.getLogger(__name__)


class InputExample:
    """
    Structure for one input example
    """
    def __init__(self, 
                 sid: str = '', 
                 src_text: str = None,
                 trg_text: str = None,  
                 label: np.ndarray = None, 
                 label_ve: np.ndarray = None,
                 lang: str = None, 
                 from_memory: bool = False,
                 ):
        """
        :param sid: id for the example
        :param src_text: the source sentence
        :param trg_text: the target sentence
        :param label: the sentence embedding of the source sentence
        :param label_ve: the vision embedding of the source image/video
        :param lang: the language of the target sentence
        :param from_memory: whether this example is sampled from the memory/buffer of previous tasks
        """
        self.sid = sid
        self.src_text = src_text
        self.trg_text = trg_text
        self.label = label
        self.label_ve = label_ve
        self.lang = lang
        self.from_memory = from_memory

    def __str__(self):
        return "<InputExample> src_text: {}, trg_text: {}, lang: {}, from_memory: {}".format(
            str(self.src_text), str(self.trg_text), str(self.lang), self.from_memory)
    
    def to_dict(self):
        return {k: getattr(self, k) for k in ['sid', 'src_text', 'trg_text', 'label', 'label_ve', 'lang', 'from_memory']}


class CLDataset(Dataset):
    def __init__(self, 
                 annotation_path: str, 
                 languages: List[str],
                 sentence_embedding_numpy_path: str,
                 vision_embedding_numpy_path: str,
                 logger: logging.Logger=None,
                 max_sentences: int = None, 
                 max_ratio: float = None,
                 sentence_embedding_cache: Union[np.ndarray, None] = None,
                 vision_embedding_cache: Union[np.ndarray, None] = None,
                 ):
        assert len(languages) == 3, f"There should be 3 languages; received {languages}"
        self.annotation_path = annotation_path
        self.languages = languages # (vision, source_language, target_language)
        self.logger = logger or global_logger

        # load embedding cache
        if sentence_embedding_cache is None:
            self.log(f'Load sentence embedding cache from {sentence_embedding_numpy_path}')
            sentence_embedding_cache = np.load(sentence_embedding_numpy_path)
        self.sentence_embedding_cache = sentence_embedding_cache
        if vision_embedding_cache is None:
            self.log(f'Load vision embedding cache from {vision_embedding_numpy_path}')
            vision_embedding_cache = np.load(vision_embedding_numpy_path)
        self.vision_embedding_cache = vision_embedding_cache

        # load annotations
        self.log(f"Load annotations from {annotation_path}")
        raw_samples = []
        with gzip.open(annotation_path, 'rt', encoding='utf8') if annotation_path.endswith('.gz') \
            else open(annotation_path, encoding='utf8') as fIn:

            for line in fIn:
                sentences = line.strip().split("\t")
                assert len(sentences) == 3, \
                    f'Each line should be a triplet: (vision_rpath, source_sentence, target_sentence); received: {sentences}'

                raw_samples.append(sentences)
                if max_sentences is not None and max_sentences > 0 and len(raw_samples) >= max_sentences:
                    break

        self.log(f"\tThere are {len(raw_samples)} lines, one of which is {raw_samples[0]}")
        
        if max_ratio is not None:
            n_samples = int(len(raw_samples) * max_ratio / 100)
            self.log(f"\tOnly keep {max_ratio}% ({n_samples}) samples")
            raw_samples = raw_samples[:n_samples]

        rpaths_set = set()
        self.rpath2index = {}
        self.samples = []
        for index, sample in enumerate(raw_samples):
            vision_rpath, source_sentence, target_sentence = sample
            if vision_rpath not in rpaths_set:
                self.rpath2index[vision_rpath] = len(rpaths_set)
                rpaths_set.add(vision_rpath)

            self.samples.append([index, vision_rpath, source_sentence, target_sentence, self.languages[-1]])
        
        if len(self.sentence_embedding_cache) < len(self.vision_embedding_cache):
            # this is because we only extract sentence embeddings of those unique sentences for cc3m
            self.log(f"\tSize of sentence embedding and vision embedding caches mismatch ({self.sentence_embedding_cache.shape} vs. {self.vision_embedding_cache.shape})")
            existed_ids_path = os.path.join(os.path.dirname(annotation_path), 'existed_ids.json')
            if not os.path.exists(existed_ids_path):
                raise FileNotFoundError(
                    f"{existed_ids_path} not found, which is required to recablibrate the size of sentence embedding cache")
            self.log(f"\tRead {existed_ids_path} to recablibrate the size of sentence embedding cache")
            existed_ids = json.load(open(existed_ids_path))
            assert len(existed_ids) == len(self.vision_embedding_cache), \
                f"length of existed_ids: {len(existed_ids)}, length of vision embedding cache: {len(self.vision_embedding_cache)}"
            self.sentence_embedding_cache = self.sentence_embedding_cache[existed_ids]
        
        if max_sentences is None and max_ratio is None:
            assert len(self.sentence_embedding_cache) == len(raw_samples), \
                    f"\tembedding_cache {len(self.sentence_embedding_cache)} lines; text data: {len(raw_samples)} lines"
        
    def log(self, msg: str):
        self.logger.info(msg)

    def next_entry(self, data_idx):
        sample = self.datasets[data_idx][self.datasets_iterator[data_idx]]

        self.datasets_iterator[data_idx] += 1
        if self.datasets_iterator[data_idx] >= len(self.datasets[data_idx]): #Restart iterator
            self.datasets_iterator[data_idx] = 0
            random.shuffle(self.datasets[data_idx])

        return sample

    def get_sentence_embedding(self, sample):
        return self.sentence_embedding_cache[sample[0]]

    def get_vision_embedding(self, sample):
        return self.vision_embedding_cache[self.rpath2index[sample[1]]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        index, vision_rpath, source_sentence, target_sentence, target_language = sample
        return InputExample(
            sid=self.rpath2index[vision_rpath],
            src_text=source_sentence, 
            trg_text=target_sentence, 
            label=self.get_sentence_embedding(sample), 
            label_ve=self.get_vision_embedding(sample),
            lang=target_language,
        )


class MemoryDataset(Dataset):
    def __init__(self, 
                 memory_size: int,
                 read_path: str = None, 
                 save_path: str = None, 
                 logger: logging.Logger = None,
                 dark_experience_replay: bool = False,
                 ) -> None:
        super().__init__()
        self.logger = logger or global_logger
        self.memory_size = memory_size
        self.memory, self.n = self.load_memory(read_path)
        self.save_path = save_path

        if len(self.memory) > self.memory_size:
            self.log(f'The size of the loaded memory is larger than your request \
                     ({len(self.memory)} vs. {self.memory_size}), truncate it now.')
            self.memory = self.memory[:self.memory_size]
        
        self.dark_experience_replay = dark_experience_replay

    def log(self, msg: str):
        self.logger.info(msg)    
    
    def __len__(self):
        return self.memory_size

    def __getitem__(self, index):
        if index >= len(self.memory):
            # the memory has not been filled
            assert len(self.memory) > 0
            index = random.randint(0, len(self.memory) - 1)
        return self.memory[index]
    
    def load_memory(self, read_path: str = None) -> Tuple[List[InputExample], int]:
        if read_path is not None:
            if MEMORY_DATASET_FN not in read_path:
                read_path = os.path.join(read_path, MEMORY_DATASET_FN)
            self.log(f'Load memory dataset from {read_path}')
            data = json.load(open(read_path))
            memory = []
            for example in data['memory']:
                kwargs = {k: (np.array(v) if 'label' in k else v) for k, v in example.items()}
                kwargs['from_memory'] = True
                memory.append(InputExample(**kwargs))
            n = data['n']
        else:
            self.log(f'Construct an empty memory dataset')
            memory, n = [], 0
        return memory, n

    def save_memory(self, save_path: str = None):
        save_path = self.save_path or save_path
        if save_path is not None:
            self.log(f'Save memory dataset of size {len(self.memory)} to {save_path}')
            os.makedirs(save_path, exist_ok=True)
            if MEMORY_DATASET_FN not in save_path:
                save_path = os.path.join(save_path, MEMORY_DATASET_FN)
            memory = []
            for example in self.memory:
                line = example.to_dict()
                line = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in line.items()}
                line['from_memory'] = True
                memory.append(line)
            with open(save_path, 'w') as wf:
                json.dump({"memory": memory, "n": self.n}, wf)

    def reservoir_sampling(self, examples: List[InputExample], labels: Tensor):
        j = 0
        for index, example in enumerate(examples):
            if example.from_memory:
                continue

            kwargs = example.to_dict()
            kwargs['from_memory'] = True
            if self.dark_experience_replay:
                # the original label is used for aligning (X_t, Y_t) bitext pairs
                # now, label is used for aligning (Y_{<t}, Y_t) bitext pairs
                kwargs['label'] = labels[index].detach().cpu().numpy()

            if len(self.memory) < self.memory_size:
                self.memory.append(InputExample(**kwargs))
            else:
                i = random.randint(0, self.n + j)
                if i < self.memory_size:
                    self.memory[i] = InputExample(**kwargs)
            j += 1
        self.n += j
    
    def language_summary(self):
        from collections import Counter
        return Counter([example.lang for example in self.memory])
