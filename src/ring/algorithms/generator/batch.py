import gc
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from tree_utils import PyTree

from ring import utils
from ring.algorithms.generator import types


def _build_batch_matrix(batchsizes: list[int]) -> jax.Array:
    arr = []
    for i, l in enumerate(batchsizes):
        arr += [i] * l
    return jnp.array(arr)


def generators_lazy(
    generators: list[types.BatchedGenerator],
    repeats: list[int],
    jit: bool = True,
) -> types.BatchedGenerator:

    batch_arr = _build_batch_matrix(repeats)
    bs_total = len(batch_arr)
    pmap, vmap = utils.distribute_batchsize(bs_total)
    batch_arr = batch_arr.reshape((pmap, vmap))

    pmap_trafo = jax.pmap
    # single GPU node, then do jit + vmap instead of pmap
    # this allows e.g. better NAN debugging capabilities
    if pmap == 1:
        pmap_trafo = lambda f: jax.jit(jax.vmap(f))
    if not jit:
        pmap_trafo = lambda f: jax.vmap(f)

    @pmap_trafo
    @jax.vmap
    def _generator(key, which_gen: int):
        return jax.lax.switch(which_gen, generators, key)

    def generator(key):
        pmap_vmap_keys = jax.random.split(key, bs_total).reshape((pmap, vmap, 2))
        data = _generator(pmap_vmap_keys, batch_arr)

        # merge pmap and vmap axis
        data = utils.merge_batchsize(data, pmap, vmap, third_dim_also=True)
        return data

    return generator


def generators_eager(
    generators: list[types.BatchedGenerator],
    n_calls: list[int],
    callback: Callable[[list[PyTree[np.ndarray]]], None],
    seed: int = 1,
    disable_tqdm: bool = False,
) -> None:

    key = jax.random.PRNGKey(seed)
    for gen, n_call in tqdm(
        zip(generators, n_calls),
        desc="executing generators",
        total=len(generators),
        disable=disable_tqdm,
    ):
        for _ in tqdm(
            range(n_call),
            desc="number of calls for each generator",
            total=n_call,
            leave=False,
            disable=disable_tqdm,
        ):
            key, consume = jax.random.split(key)
            sample = gen(consume)
            # converts also to numpy; but with np.array.flags.writeable = False
            sample = jax.device_get(sample)
            # this then sets this flag to True
            sample = jax.tree_map(np.array, sample)

            sample_flat, _ = jax.tree_util.tree_flatten(sample)
            size = 1 if len(sample_flat) == 0 else sample_flat[0].shape[0]
            callback([jax.tree_map(lambda a: a[i].copy(), sample) for i in range(size)])

            # cleanup
            del sample, sample_flat
            gc.collect()
