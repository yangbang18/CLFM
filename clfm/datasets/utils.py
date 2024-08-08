import numpy as np
from typing import List
from collections import defaultdict
from joblib import Parallel, delayed
from .ConcatDataset import ConcatDataset


def prepare_token_mapping(tokenizer, 
                          concat_datasets: List[ConcatDataset], 
                          exclude_special_tokens: bool = False, 
                          n_jobs: int = 8,
                          ignore_memory_dataset: bool = False,
                          return_token_counters: bool = False,
                          ):
    final_mapping = defaultdict(set)
    final_counters = defaultdict(int)

    def _run(samples):
        token_mapping = defaultdict(set)
        token_counters = defaultdict(int)
        for sample in samples:
            if isinstance(sample, (list, tuple)):
                index, vision_rpath, source_sentence, target_sentence, target_language = sample
            else:
                target_sentence, target_language = sample.trg_text, sample.lang
            for token_id in tokenizer(target_sentence, add_special_tokens=not exclude_special_tokens)['input_ids']:
                token_mapping[token_id].add(target_language)
                token_counters[token_id] += 1
        return token_mapping, token_counters

    def _update(token_mapping, token_counters):
        for k, v in token_mapping.items():
            final_mapping[k] = final_mapping[k] | v
        for k, v in token_counters.items():
            final_counters[k] += v

    for concat_dataset in concat_datasets:
        for dataset in concat_dataset.datasets:
            assert hasattr(dataset, 'memory') ^ hasattr(dataset, 'samples')

            if hasattr(dataset, 'memory') and ignore_memory_dataset:
                continue

            if hasattr(dataset, 'memory'):
                samples = dataset.memory
                if len(samples) == 0:
                    continue
            else:
                samples = dataset.samples
            
            this_n_jobs = min(len(samples), n_jobs)

            for (token_mapping, token_counters) in Parallel(n_jobs=this_n_jobs)(delayed(_run)(split.tolist()) for split in np.array_split(samples, this_n_jobs)):
                _update(token_mapping, token_counters)

    final_mapping = {k: list(v) for k, v in final_mapping.items()}
    if return_token_counters:
        return final_mapping, final_counters
    return final_mapping
