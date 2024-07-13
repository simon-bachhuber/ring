from pathlib import Path
import random
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import tree_utils

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


def generators_eager_to_list(
    generators: list[types.BatchedGenerator],
    n_calls: list[int],
    seed: int = 1,
    disable_tqdm: bool = False,
) -> list[tree_utils.PyTree]:

    key = jax.random.PRNGKey(seed)
    data = []
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
            data.extend([jax.tree_map(lambda a: a[i], sample) for i in range(size)])

    return data


def _is_nan(ele: tree_utils.PyTree, i: int, verbose: bool = False):
    isnan = np.any([np.any(np.isnan(arr)) for arr in jax.tree_util.tree_leaves(ele)])
    if isnan:
        X, y = ele
        dt = X["dt"].flatten()[0]
        if verbose:
            print(f"Sample with idx={i} is nan. It will be replaced. (dt={dt})")
        return True
    return False


def _replace_elements_w_nans(list_of_data: list, include_samples: list[int]) -> list:
    list_of_data_nonan = []
    for i, ele in enumerate(list_of_data):
        if _is_nan(ele, i, verbose=True):
            while True:
                j = random.choice(include_samples)
                if not _is_nan(list_of_data[j], j):
                    ele = list_of_data[j]
                    break
        list_of_data_nonan.append(ele)
    return list_of_data_nonan


_list_of_data = None
_paths = None


def _data_fn_from_paths(
    paths: list[str],
    include_samples: list[int] | None,
    load_all_into_memory: bool,
    tree_transform,
):
    "`data_fn` returns numpy arrays."
    global _list_of_data, _paths

    # expanduser
    paths = [utils.parse_path(p, mkdir=False) for p in paths]
    extensions = list(set([Path(p).suffix for p in paths]))
    assert len(extensions) == 1, f"{extensions}"
    h5 = extensions[0] == ".h5"

    if h5 and not load_all_into_memory:

        def data_fn(indices: list[int]):
            tree = utils.hdf5_load_from_multiple(paths, indices)
            return tree if tree_transform is None else tree_transform(tree)

        N = sum([utils.hdf5_load_length(p) for p in paths])
    else:

        load_from_path = utils.hdf5_load if h5 else utils.pickle_load

        def load_fn(path):
            tree = load_from_path(path)
            tree = tree if tree_transform is None else tree_transform(tree)
            return [
                jax.tree_map(lambda arr: arr[i], tree)
                for i in range(tree_utils.tree_shape(tree))
            ]

        if paths != _paths or len(_list_of_data) == 0:
            _paths = paths

            _list_of_data = []
            for p in paths:
                _list_of_data += load_fn(p)

        N = len(_list_of_data)
        list_of_data = _replace_elements_w_nans(
            _list_of_data,
            include_samples if include_samples is not None else list(range(N)),
        )

        if include_samples is not None:
            list_of_data = [
                ele if i in include_samples else None
                for i, ele in enumerate(list_of_data)
            ]

        def data_fn(indices: list[int]):
            return tree_utils.tree_batch(
                [list_of_data[i] for i in indices], backend="numpy"
            )

    if include_samples is None:
        include_samples = list(range(N))

    return data_fn, include_samples.copy()


def generator_from_data_fn(
    data_fn,
    include_samples: list[int],
    shuffle: bool,
    batchsize: int,
) -> types.BatchedGenerator:
    # such that we don't mutate out of scope
    include_samples = include_samples.copy()

    N = len(include_samples)
    n_batches, i = N // batchsize, 0

    def generator(key: jax.Array):
        nonlocal i
        if shuffle and i == 0:
            random.shuffle(include_samples)

        start, stop = i * batchsize, (i + 1) * batchsize
        batch = data_fn(include_samples[start:stop])

        i = (i + 1) % n_batches
        return utils.pytree_deepcopy(batch)

    return generator


def generator_from_paths(
    paths: list[str],
    batchsize: int,
    include_samples: Optional[list[int]] = None,
    shuffle: bool = True,
    load_all_into_memory: bool = False,
    tree_transform=None,
) -> tuple[types.BatchedGenerator, int]:
    "Returns: gen, where gen(key) -> Pytree[numpy]"
    data_fn, include_samples = _data_fn_from_paths(
        paths, include_samples, load_all_into_memory, tree_transform
    )

    N = len(include_samples)
    assert N >= batchsize

    generator = generator_from_data_fn(data_fn, include_samples, shuffle, batchsize)

    return generator, N
