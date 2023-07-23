from typing import Optional, Tuple

import jax
from tree_utils import PyTree


def distribute_batchsize(batchsize: int) -> Tuple[int, int]:
    """Distributes batchsize accross pmap and vmap."""
    vmap_size_min = 8
    if batchsize <= vmap_size_min:
        return 1, batchsize
    else:
        n_devices = jax.local_device_count()
        assert (
            batchsize % n_devices
        ) == 0, f"Your GPU count of {n_devices} does not split batchsize {batchsize}"
        vmap_size = int(batchsize / n_devices)
        return int(batchsize / vmap_size), vmap_size


def merge_batchsize(tree: PyTree, pmap_size: int, vmap_size: int) -> PyTree:
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


CPU_ONLY = False


def backend(cpu_only: bool = False, n_gpus: Optional[int] = None):
    "Sets backend for all jax operations (including this library)."
    global CPU_ONLY

    if cpu_only and not CPU_ONLY:
        CPU_ONLY = True
        from jax import config

        config.update("jax_platform_name", "cpu")
