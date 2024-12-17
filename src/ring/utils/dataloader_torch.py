import os
from typing import Any, Optional
import warnings

import jax
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tree_utils import PyTree

from ring.utils import parse_path
from ring.utils import pickle_load


class FolderOfFilesDataset(Dataset):
    def __init__(self, path, transform=None):
        self.files = self.listdir(path)
        self.transform = transform
        self.N = len(self.files)

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        element = self._load_file(self.files[idx])
        if self.transform is not None:
            element = self.transform(element)
        return element

    @staticmethod
    def listdir(path: str) -> list:
        return [parse_path(path, file) for file in os.listdir(path)]

    @staticmethod
    def _load_file(file_path: str) -> Any:
        return pickle_load(file_path)


def dataset_to_generator(
    dataset: Dataset,
    batch_size: int,
    shuffle=True,
    seed: int = 1,
    num_workers: Optional[int] = None,
    **kwargs,
):
    torch.manual_seed(seed)

    if num_workers is None:
        num_workers = _get_number_of_logical_cores()

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        num_workers=num_workers,
        **kwargs,
    )
    dl_iter = iter(dl)

    def to_numpy(tree: PyTree[torch.Tensor]):
        return jax.tree_map(lambda tensor: tensor.numpy(), tree)

    def generator(_):
        nonlocal dl, dl_iter
        try:
            return to_numpy(next(dl_iter))
        except StopIteration:
            dl_iter = iter(dl)
            return to_numpy(next(dl_iter))

    return generator


def _get_number_of_logical_cores() -> int:
    N = None
    if hasattr(os, "sched_getaffinity"):
        try:
            N = len(os.sched_getaffinity(0))
        except Exception:
            pass
    if N is None:
        N = os.cpu_count()
    if N is None:
        warnings.warn(
            "Could not automatically set the `num_workers` variable, defaults to `0`"
        )
        N = 0
    return N


class MultiDataset(Dataset):
    def __init__(self, datasets, transform=None):
        """
        Args:
            datasets: A list of datasets to sample from.
            transform: A function that takes N items (one from each dataset) and combines them.
        """  # noqa: E501
        self.datasets = datasets
        self.transform = transform

    def __len__(self):
        # Length is defined by the smallest dataset in the list
        return min(len(ds) for ds in self.datasets)

    def __getitem__(self, idx):
        sampled_items = [ds[idx] for ds in self.datasets]

        if self.transform:
            # Apply the transformation to all sampled items
            return self.transform(*sampled_items)

        return tuple(sampled_items)


class ShuffledDataset(Dataset):
    def __init__(self, dataset):
        """
        Wrapper that shuffles the dataset indices once.

        Args:
            dataset (Dataset): The original dataset to shuffle.
        """
        self.dataset = dataset
        self.shuffled_indices = np.random.permutation(
            len(dataset)
        )  # Shuffle indices once

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns the data at the shuffled index.

        Args:
            idx (int): Index in the shuffled dataset.
        """
        original_idx = self.shuffled_indices[idx]
        return self.dataset[original_idx]


def dataset_to_Xy(ds: Dataset):
    return dataset_to_generator(ds, batch_size=len(ds), shuffle=False, num_workers=0)(
        None
    )
