from pathlib import Path
import random
from typing import Callable, Optional
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import tree_utils
from tree_utils import PyTree
from tree_utils import tree_batch

from ... import utils
from .types import BatchedGenerator
from .types import Generator


def _build_batch_matrix(batchsizes: list[int]) -> jax.Array:
    arr = []
    for i, l in enumerate(batchsizes):
        arr += [i] * l
    return jnp.array(arr)


def batch_generators_lazy(
    generators: Generator | list[Generator],
    batchsizes: int | list[int] = 1,
) -> BatchedGenerator:
    """Create a large generator by stacking multiple generators lazily."""
    generators = utils.to_list(generators)

    generators, batchsizes = _process_sizes_batchsizes_generators(
        generators, batchsizes
    )

    batch_arr = _build_batch_matrix(batchsizes)
    bs_total = len(batch_arr)
    pmap, vmap = utils.distribute_batchsize(bs_total)
    batch_arr = batch_arr.reshape((pmap, vmap))

    pmap_trafo = jax.pmap
    # single GPU node, then do jit + vmap instead of pmap
    # this allows e.g. better NAN debugging capabilities
    if pmap == 1:
        pmap_trafo = lambda f: jax.jit(jax.vmap(f))

    @pmap_trafo
    @jax.vmap
    def _generator(key, which_gen: int):
        return jax.lax.switch(which_gen, generators, key)

    def generator(key):
        pmap_vmap_keys = jax.random.split(key, bs_total).reshape((pmap, vmap, 2))
        data = _generator(pmap_vmap_keys, batch_arr)

        # merge pmap and vmap axis
        data = utils.merge_batchsize(data, pmap, vmap)
        return data

    return generator


def batch_generators_eager_to_list(
    generators: Generator | list[Generator],
    sizes: int | list[int],
    seed: int = 1,
) -> list[tree_utils.PyTree]:
    "Returns list of unbatched sequences."
    generators, sizes = _process_sizes_batchsizes_generators(generators, sizes)

    key = jax.random.PRNGKey(seed)
    data = []
    for gen, size in tqdm(zip(generators, sizes), desc="eager data generation"):
        key, consume = jax.random.split(key)
        sample = batch_generators_lazy(gen, size)(consume)
        data.extend([jax.tree_map(lambda a: a[i], sample) for i in range(size)])
    return data


def _data_fn_from_paths(
    paths: list[str], include_samples: list[int] | None, load_all_into_memory: bool
):
    "`data_fn` returns numpy arrays."
    # expanduser
    paths = [utils.parse_path(p, mkdir=False) for p in paths]

    extensions = list(set([Path(p).suffix for p in paths]))
    assert len(extensions) == 1

    if extensions[0] == ".h5" and not load_all_into_memory:
        N = sum([utils.hdf5_load_length(p) for p in paths])

        def data_fn(indices: list[int]):
            return utils.hdf5_load_from_multiple(paths, indices)

    else:
        # TODO
        from x_xy.subpkgs import ml

        if extensions[0] == ".h5":

            def load_fn(path):
                tree = utils.hdf5_load(path)
                return [
                    jax.tree_map(lambda arr: arr[i], tree)
                    for i in range(tree_utils.tree_shape(tree))
                ]

        else:
            load_fn = ml.load

        list_of_data = []
        for p in paths:
            list_of_data += load_fn(p)

        N = len(list_of_data)
        if include_samples is not None:
            list_of_data = [
                ele if i in include_samples else None
                for i, ele in enumerate(list_of_data)
            ]

        def data_fn(indices: list[int]):
            return tree_batch([list_of_data[i] for i in indices], backend="numpy")

    if include_samples is None:
        include_samples = list(range(N))
    else:
        # safety copy; we shuffle it in the next function
        include_samples = include_samples.copy()

    return data_fn, include_samples


def _generator_from_data_fn(
    data_fn,
    include_samples: list[int],
    output_transform,
    shuffle: bool,
    batchsize: int,
):
    N = len(include_samples)
    n_batches, i = N // batchsize, 0

    def generator(key: jax.Array):
        nonlocal i
        if shuffle and i == 0:
            random.shuffle(include_samples)

        start, stop = i * batchsize, (i + 1) * batchsize
        batch = data_fn(include_samples[start:stop])
        batch = batch if output_transform is None else output_transform(batch)

        i = (i + 1) % n_batches
        return batch

    return generator


def batched_generator_from_paths(
    paths: list[str],
    batchsize: int,
    include_samples: Optional[list[int]] = None,
    shuffle: bool = True,
    output_transform: Optional[
        Callable[[PyTree[np.ndarray]], PyTree[np.ndarray]]
    ] = None,
    load_all_into_memory: bool = False,
):
    "Returns: gen, where gen(key) -> Pytree[numpy]"
    data_fn, include_samples = _data_fn_from_paths(
        paths,
        include_samples,
        load_all_into_memory,
    )

    N = len(include_samples)
    assert N >= batchsize

    generator = _generator_from_data_fn(
        data_fn, include_samples, output_transform, shuffle, batchsize
    )

    return generator, N


def batched_generator_from_list(
    data: list,
    batchsize: int,
    shuffle: bool = True,
    drop_last: bool = True,
    seed: int = 1,
    # output_transform: Optional[Callable[[jax.Array, PyTree], PyTree]] = None,
) -> BatchedGenerator:
    assert drop_last, "Not `drop_last` is currently not implemented."
    assert len(data) >= batchsize

    N, i = len(data) // batchsize, 0
    random.seed(seed)

    def generator(key: jax.Array):
        nonlocal i
        if shuffle and i == 0:
            random.shuffle(data)

        start, stop = i * batchsize, (i + 1) * batchsize
        batch = tree_batch(data[start:stop])
        # batch = batch if output_transform is None else output_transform(key, batch)

        i = (i + 1) % N
        return batch

    return generator


def batch_generators_eager(
    generators: Generator | list[Generator],
    sizes: int | list[int],
    batchsize: int,
    shuffle: bool = True,
    drop_last: bool = True,
    seed: int = 1,
) -> BatchedGenerator:
    """Eagerly create a large precomputed generator by calling multiple generators
    and stacking their output."""

    data = batch_generators_eager_to_list(generators, sizes, seed=seed)
    # currently still on device; copy to host / numpy
    data = jax.device_get(data)
    return batched_generator_from_list(data, batchsize, shuffle, drop_last, seed=seed)


def _process_sizes_batchsizes_generators(
    generators: Generator | list[Generator],
    batchsizes_or_sizes: int | list[int],
) -> tuple[list, list]:
    generators = utils.to_list(generators)
    assert len(generators) > 0, "No generator was passed."

    if isinstance(batchsizes_or_sizes, int):
        assert (
            batchsizes_or_sizes // len(generators)
        ) > 0, f"Batchsize or size too small. {batchsizes_or_sizes} < {len(generators)}"
        list_sizes = len(generators) * [batchsizes_or_sizes // len(generators)]
    else:
        list_sizes = batchsizes_or_sizes
        assert 0 not in list_sizes

    assert len(generators) == len(list_sizes)

    _WARN_SIZE = 4096
    for size in list_sizes:
        if size >= _WARN_SIZE:
            warnings.warn(
                f"A generator will be called with a large batchsize of {size} "
                f"(warn limit is {_WARN_SIZE}). The generator sizes are {list_sizes}."
            )
            break

    return generators, list_sizes
