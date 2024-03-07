"""Save/load pytrees to disk. Allows for
- partial loading of hdf5 files (only certain batch indices are loaded in memory).
(taken and modified from https://gist.github.com/nirum/b119bbbd32d22facee3071210e08ecdf)
"""

import collections
from functools import partial
import os
from pathlib import Path
from typing import Optional

from flax import struct
import h5py
import jax
import numpy as np

hdf5_extension = "h5"


def save(filepath: str, tree, overwrite: bool = False):
    """Saves a pytree to an hdf5 file.

    Args:
      filepath: str, Path of the hdf5 file to create.
      tree: pytree, Recursive collection of tuples, lists, dicts,
        namedtuples and numpy arrays to store.
    """
    filepath = _parse_path(filepath, hdf5_extension, overwrite)
    with h5py.File(filepath, "w") as f:
        # jax.device_get converts to numpy array
        _savetree(jax.device_get(tree), f, "pytree")


def load(
    filepath: str,
    indices: Optional[int | list[int] | slice] = None,
    axis: int = 0,
):
    """Loads a pytree from an hdf5 file.

    Args:
      filepath: str, Path of the hdf5 file to load.
      indices: if not `None`, take only these indices of the leaf array values
        along `axis`. Note that this truly only loads those indices into RAM.
      axis: int, axis along which to take indices, usually a batch axis.
    """

    filepath = _parse_path(filepath, hdf5_extension)

    with h5py.File(filepath, "r") as f:
        return _loadtree(f["pytree"], indices, axis)


def _call_fn(fn):
    return fn()


def load_from_multiple(filepaths: list[str], indices: list[int]):
    assert len(filepaths) > 1

    borders = np.cumsum([load_length(fp) for fp in filepaths])
    indices = np.sort(indices)
    belongs_to = np.searchsorted(borders - 1, indices)

    assert indices[-1] < borders[-1]

    borders = np.concatenate((np.array([0]), borders))
    loaders = []
    for i, fp in enumerate(filepaths):
        indices_fp = list(indices[belongs_to == i] - borders[i])
        if len(indices_fp) == 0:
            continue
        loaders.append(partial(load, fp, indices_fp))

    trees = [loader() for loader in loaders]

    return _tree_concat(trees)


@struct.dataclass
class _Shape:
    shape: tuple


def load_length(filepath: str, axis: int = 0) -> int:
    """Loads the length of an undefined leaf along an axis.

    Args:
        filepath (str): str, Path of the hdf5 file to load.
        axis (int, optional): Axis to get the length along. Defaults to 0.

    Returns:
        int: Lenght of that axis dimensionality.
    """
    filepath = _parse_path(filepath, hdf5_extension)

    with h5py.File(filepath, "r") as f:
        tree_of_shapes = _lazy_tree_map(lambda leaf: _Shape(leaf.shape), f["pytree"])
        return jax.tree_util.tree_flatten(
            tree_of_shapes, is_leaf=lambda leaf: isinstance(leaf, _Shape)
        )[0][0].shape[axis]


def _parse_path(
    path: str,
    extension: Optional[str] = None,
    file_exists_ok: bool = True,
) -> str:
    path = Path(os.path.expanduser(path))

    if extension is not None:
        if extension != "":
            extension = ("." + extension) if (extension[0] != ".") else extension
        path = path.with_suffix(extension)

    if not file_exists_ok and os.path.exists(path):
        raise Exception(f"File {path} already exists but shouldn't")

    return str(path)


def _tree_concat(trees: list):
    # otherwise scalar-arrays will lead to indexing error
    trees = jax.tree_map(lambda arr: np.atleast_1d(arr), trees)

    if len(trees) == 0:
        return trees
    if len(trees) == 1:
        return trees[0]

    return jax.tree_util.tree_map(lambda *arrs: np.concatenate(arrs, axis=0), *trees)


def _is_namedtuple(x):
    """Duck typing check if x is a namedtuple."""
    return isinstance(x, tuple) and getattr(x, "_fields", None) is not None


def _savetree(tree, group, name):
    """Recursively save a pytree to an h5 file group."""

    if isinstance(tree, np.ndarray):
        group.create_dataset(name, data=tree)

    else:
        subgroup = group.create_group(name)
        subgroup.attrs["type"] = type(tree).__name__

        if _is_namedtuple(tree):
            for k, subtree in tree._asdict().items():
                _savetree(subtree, subgroup, k)
        elif isinstance(tree, tuple) or isinstance(tree, list):
            for k, subtree in enumerate(tree):
                _savetree(subtree, subgroup, f"arr{k}")
        elif isinstance(tree, dict):
            for k, subtree in tree.items():
                _savetree(subtree, subgroup, k)
        else:
            raise ValueError(f"Unrecognized type {type(tree)}")


def _loadtree(tree, indices: int | list[int] | slice | None, axis: int):
    """Recursively load a pytree from an h5 file group."""

    if indices is None:
        return _lazy_tree_map(lambda leaf: np.asarray(leaf), tree)

    if isinstance(indices, list):
        # must be in increasing order for h5py
        indices = sorted(indices)

    def func(leaf):
        shape = leaf.shape
        selection = [slice(None)] * len(shape)
        selection[axis] = indices
        # convert list to tuple; otherwise it errors
        selection = tuple(selection)
        return np.asarray(leaf[selection])

    return _lazy_tree_map(func, tree)


def _lazy_tree_map(func, leaf):
    if isinstance(leaf, h5py.Dataset):
        return func(leaf)

    else:
        leaf_type = leaf.attrs["type"]
        values = map(lambda leaf: _lazy_tree_map(func, leaf), leaf.values())

        if leaf_type == "dict":
            return dict(zip(leaf.keys(), values))
        elif leaf_type == "list":
            return list(values)
        elif leaf_type == "tuple":
            return tuple(values)
        else:  # namedtuple
            return collections.namedtuple(leaf_type, leaf.keys())(*values)
