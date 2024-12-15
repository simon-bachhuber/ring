import os
from typing import Optional
import warnings

import jax
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tree_utils import PyTree

from ring.utils import parse_path
from ring.utils import pickle_load


class FolderOfPickleFilesDataset(Dataset):
    def __init__(self, path, transform=None):
        self.files = self.listdir(path)
        self.transform = transform
        self.N = len(self.files)

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        element = pickle_load(self.files[idx])
        if self.transform is not None:
            element = self.transform(element)
        return element

    @staticmethod
    def listdir(path: str) -> list:
        return [parse_path(path, file) for file in os.listdir(path)]


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
