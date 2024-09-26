import os
import random
from typing import Callable, Optional

import jax
import numpy as np
from ring.utils import parse_path
from ring.utils import pickle_load
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import tqdm
from tree_utils import PyTree


def make_generator(
    *paths,
    batch_size,
    transform,
    shuffle=True,
    seed: int = 1,
    backend: str = "eager",
    **kwargs,
):
    if backend == "grain":
        _make_gen = pygrain_generator
    elif backend == "torch":
        _make_gen = pytorch_generator
    elif backend == "eager":
        _make_gen = eager_generator
    else:
        raise NotImplementedError

    return _make_gen(
        *paths,
        batch_size=batch_size,
        transform=transform,
        shuffle=shuffle,
        seed=seed,
        **kwargs,
    )


T = PyTree[np.ndarray]


class _Dataset(Dataset):
    def __init__(self, *paths, transform):

        self.files = [self.listdir(path) for path in paths]
        Ns = set([len(f) for f in self.files])
        assert len(Ns) == 1, f"{Ns}"

        self.P = len(self.files)
        self.N = list(Ns)[0]
        self.transform = transform

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        element = [pickle_load(self.files[p][idx]) for p in range(self.P)]
        if self.transform is not None:
            element = self.transform(element)
        return element

    @staticmethod
    def listdir(path: str) -> list:
        return [parse_path(path, file) for file in os.listdir(path)]

    def __call__(self, idx: int):
        return self[idx]


class TransformTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, element):
        if self.transform is None:
            return element
        return self.transform(element, np.random.default_rng())


def pytorch_generator(
    *paths,
    batch_size: int,
    transform: Optional[Callable[[T], T]] = None,
    shuffle=True,
    seed: int = 1,
    **kwargs,
):
    torch.manual_seed(seed)

    ds = _Dataset(*paths, transform=TransformTransform(transform))
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        multiprocessing_context="spawn" if kwargs.get("num_workers", 0) > 0 else None,
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


def eager_generator(
    *paths,
    batch_size: int,
    transform: Optional[Callable[[T], T]] = None,
    shuffle=True,
    seed=1,
):
    from ring import RCMG

    random.seed(seed)

    ds = _Dataset(*paths, transform=TransformTransform(transform))
    data = [ds[i] for i in tqdm.tqdm(range(len(ds)), total=len(ds))]
    return RCMG.eager_gen_from_list(data, batch_size, shuffle=shuffle)


def pygrain_generator(
    *paths, batch_size: int, transform=None, shuffle=True, seed=1, **kwargs
):

    import grain.python as pygrain  # type: ignore

    class _Transform(pygrain.RandomMapTransform):
        def random_map(self, element, rng: np.random.Generator):
            return transform(element, rng)

    ds = _Dataset(*paths, transform=None)
    dl = pygrain.load(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        transformations=[_Transform()],
        **kwargs,
    )
    iter_dl = iter(dl)

    def generator(_):
        return next(iter_dl)

    return generator
