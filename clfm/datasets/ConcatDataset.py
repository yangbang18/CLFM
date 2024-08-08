import random
import logging
import numpy as np
from sentence_transformers import LoggingHandler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, get_worker_info
from torch.utils.data import ConcatDataset as TorchConcatDataset
from typing import Iterable, Union, List


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

global_logger = logging.getLogger(__name__)


class ConcatDataset(TorchConcatDataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
        balanced_sampling (bool): if True, iterate datasets to yield samples, so that a batch will contain equal number of samples from each dataset
    """
    def __init__(self, 
                 datasets: Iterable[Dataset], 
                 balanced_sampling: bool = False, 
                 experience_replay: bool = False, 
                 num_workers: int = 0,
                 ) -> None:
        super().__init__(datasets)
        self.balanced_sampling = balanced_sampling
        self.experience_replay = experience_replay
        
        self.is_the_first_task = False
        if self.experience_replay:
            assert self.balanced_sampling
            assert len(datasets) == 2, \
                "for experience replay, you should define a dataset for the current task and a memory dataset for previous tasks"
            assert hasattr(datasets[1], 'memory'), "the second dataset should be a memory dataset"
            if len(datasets[1].memory) == 0:
                self.is_the_first_task = True
        
        if self.balanced_sampling:
            if num_workers == 0:
                self.dataset_pointer = 0
                self.num_visits = [0] * len(datasets)
                self.sample_ids = []
                for dataset in datasets:
                    ids = [_ for _ in range(len(dataset))]
                    random.shuffle(ids)
                    self.sample_ids.append(ids)
            else:
                if self.is_the_first_task:
                    self.dataset_pointer = [0] * num_workers
                else:
                    self.dataset_pointer = [_ for _ in range(num_workers)]
                self.num_visits = [[0] * len(datasets) for _ in range(num_workers)]
                self.sample_ids = [[] for _ in range(num_workers)]
                for dataset in datasets:
                    ids = [_ for _ in range(len(dataset))]
                    random.shuffle(ids)
                    splits = np.array_split(ids, num_workers)
                    for i, split in enumerate(splits):
                        self.sample_ids[i].append(split.tolist())
    
    def __len__(self):
        if self.experience_replay:
            if self.is_the_first_task:
                # for the first task, we do not apply experience replay
                return len(self.datasets[0])
            else:
                # to ensure that all samples of the current task can be yielded once per epoch
                return len(self.datasets[0]) * 2
        return super().__len__()

    def __getitem__(self, idx):
        if not self.balanced_sampling:
            return super().__getitem__(idx)

        worker_info = get_worker_info()
        if worker_info is not None:
            return self.__getitem_multiple_workers__(worker_info.id)

        dataset_idx = self.dataset_pointer % len(self.datasets)
        num_visit = self.num_visits[dataset_idx]
        sample_idx = self.sample_ids[dataset_idx][num_visit]

        self.num_visits[dataset_idx] += 1
        if self.num_visits[dataset_idx] == len(self.datasets[dataset_idx]):
            self.num_visits[dataset_idx] = 0
            random.shuffle(self.sample_ids[dataset_idx])
        
        self.dataset_pointer += 1
        if self.experience_replay and self.is_the_first_task:
            self.dataset_pointer = 0 # always get examples from the first dataset

        example = self.datasets[dataset_idx][sample_idx]
        return example

    def __getitem_multiple_workers__(self, worker_id):
        dataset_idx = self.dataset_pointer[worker_id] % len(self.datasets)
        num_visit = self.num_visits[worker_id][dataset_idx]
        sample_idx = self.sample_ids[worker_id][dataset_idx][num_visit]

        self.num_visits[worker_id][dataset_idx] += 1
        if self.num_visits[worker_id][dataset_idx] == len(self.sample_ids[worker_id][dataset_idx]):
            self.num_visits[worker_id][dataset_idx] = 0
            random.shuffle(self.sample_ids[worker_id][dataset_idx])
        
        self.dataset_pointer[worker_id] += 1
        if self.experience_replay and self.is_the_first_task:
            self.dataset_pointer[worker_id] = 0 # always get examples from the first dataset

        example = self.datasets[dataset_idx][sample_idx]
        return example

    def update_memory(self, examples, labels):
        assert self.experience_replay
        memory_dataset = self.datasets[1]
        memory_dataset.reservoir_sampling(examples, labels)


def get_concat_dataset_and_loader(
        datasets: List[Dataset],
        weights: Union[None, List[int]],
        batch_size: int,
        num_workers: int,
        is_train: bool = True,
        num_samples: int = None,
        balanced_sampling: bool = False,
        experience_replay: bool = False,
        double_batch_size: bool = False,
    ):
    if experience_replay:
        assert balanced_sampling
        assert len(datasets) == 2

    concat_dataset = ConcatDataset(datasets, balanced_sampling, experience_replay, num_workers)

    if experience_replay and not concat_dataset.is_the_first_task and double_batch_size:
        # ensure that the number of training steps is same as the method without experience replay
        batch_size = batch_size * 2

    if weights is not None and len(weights) > 1:
        assert len(weights) == len(datasets)
        assert is_train is False

        samples_weights = []
        for dataset, dataset_weight in zip(datasets, weights):
            samples_weights += [dataset_weight] * len(dataset)
        
        sampler = WeightedRandomSampler(
            samples_weights, 
            num_samples=num_samples or len(concat_dataset),
            replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = is_train

    loader = DataLoader(
        concat_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=shuffle, 
        drop_last=is_train,
        sampler=sampler,
    )
    return concat_dataset, loader
