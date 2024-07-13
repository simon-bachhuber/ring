from typing import Tuple

import jax
from tree_utils import PyTree


def batchsize_thresholds():
    backend = jax.default_backend()
    if backend == "cpu":
        vmap_size_min = 1
        eager_threshold = 4
    elif backend == "gpu":
        vmap_size_min = 8
        eager_threshold = 32
    else:
        raise Exception(
            f"Backend {backend} has no default values, please add them in this function"
        )
    return vmap_size_min, eager_threshold


def distribute_batchsize(batchsize: int) -> Tuple[int, int]:
    """Distributes batchsize accross pmap and vmap."""
    vmap_size_min = batchsize_thresholds()[0]
    if batchsize <= vmap_size_min:
        return 1, batchsize
    else:
        n_devices = jax.local_device_count()
        msg = (
            f"Your local device count of {n_devices} does not split batchsize"
            + f" {batchsize}. local devices are {jax.local_devices()}"
        )
        assert (batchsize % n_devices) == 0, msg
        vmap_size = int(batchsize / n_devices)
        return int(batchsize / vmap_size), vmap_size


def merge_batchsize(
    tree: PyTree, pmap_size: int, vmap_size: int, third_dim_also: bool = False
) -> PyTree:
    if third_dim_also:
        return jax.tree_map(
            lambda arr: arr.reshape(
                (pmap_size * vmap_size * arr.shape[2],) + arr.shape[3:]
            ),
            tree,
        )
    return jax.tree_map(
        lambda arr: arr.reshape((pmap_size * vmap_size,) + arr.shape[2:]), tree
    )


def expand_batchsize(tree: PyTree, pmap_size: int, vmap_size: int) -> PyTree:
    return jax.tree_map(
        lambda arr: arr.reshape(
            (
                pmap_size,
                vmap_size,
            )
            + arr.shape[1:]
        ),
        tree,
    )
