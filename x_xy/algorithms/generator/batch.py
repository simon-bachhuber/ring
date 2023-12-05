import math
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


def _size_per_device_sweetspot() -> int:
    backend = jax.default_backend()
    if backend == "cpu":
        return 4
    elif backend == "gpu":
        return 64
    else:
        raise NotImplementedError(f"The backend '{backend}' is not 'cpu' nor 'gpu'.")


def _single_call_optimized(
    generators: Generator | list[Generator],
    batchsizes: int | list[int] = 1,
    stochastic: bool = False,
):
    # currently supports only batchsizes being int, not list[int]
    if isinstance(batchsizes, list):
        return batch_generators_lazy(
            generators=generators,
            batchsizes=batchsizes,
            stochastic=stochastic,
            single_call_opt=False,
        )

    size = batchsizes
    size_sweetspot = (
        jax.device_count(jax.default_backend()) * _size_per_device_sweetspot()
    )
    size_sweetspot = min(size, size_sweetspot)

    gen_sweetspot = batch_generators_lazy(
        generators=generators,
        batchsizes=size_sweetspot,
        stochastic=stochastic,
        single_call_opt=False,
    )
    n_calls = math.ceil(size / size_sweetspot)

    def forloop_generator(key):
        keys = jax.random.split(key, n_calls + 1)

        data = []
        for key in keys[:-1]:
            data.append(gen_sweetspot(key))

        # permute the data from last call because it might be not used completely
        # and then we wouldn't get a uniform sample from the generators as we
        # exepect to get
        data[-1] = jax.tree_map(
            lambda arr: jax.random.permutation(keys[-1], arr), data[-1]
        )

        data = tree_utils.tree_batch(data, True, "jax")
        return tree_utils.tree_slice(data, start=0, slice_size=size)

    return forloop_generator


def batch_generators_lazy(
    generators: Generator | list[Generator],
    batchsizes: int | list[int] = 1,
    stochastic: bool = False,
    single_call_opt: bool = False,
) -> BatchedGenerator:
    """Create a large generator by stacking multiple generators lazily.
    NOTE: If `stochastic` then `batchsizes` must be a single integer.
    """

    if single_call_opt:
        warnings.warn(
            "Unfortunately, the flag `single_call_opt` seems to always"
            " decrease performance."
        )
        return _single_call_optimized(
            generators=generators, batchsizes=batchsizes, stochastic=stochastic
        )

    generators = utils.to_list(generators)

    if stochastic:
        assert isinstance(batchsizes, int)
        bs_total = batchsizes
        pmap, vmap = utils.distribute_batchsize(bs_total)
    else:
        generators, batchsizes = _process_sizes_batchsizes_generators(
            generators, batchsizes
        )

        batch_arr_nonstoch = _build_batch_matrix(batchsizes)
        bs_total = len(batch_arr_nonstoch)
        pmap, vmap = utils.distribute_batchsize(bs_total)
        batch_arr_nonstoch = batch_arr_nonstoch.reshape((pmap, vmap))

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
        if stochastic:
            key, consume = jax.random.split(key)
            batch_arr = jax.random.choice(
                consume, jnp.arange(len(generators)), shape=(pmap, vmap)
            )
        else:
            batch_arr = batch_arr_nonstoch

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
    transfer_to_cpu: bool = True,
) -> list[tree_utils.PyTree]:
    "Returns list of unbatched sequences."
    generators, sizes = _process_sizes_batchsizes_generators(generators, sizes)

    key = jax.random.PRNGKey(seed)
    data = []
    for gen, size in tqdm(zip(generators, sizes), desc="eager data generation"):
        key, consume = jax.random.split(key)
        sample = batch_generators_lazy(gen, size)(consume)
        if transfer_to_cpu:
            sample = jax.device_put(sample, jax.devices("cpu")[0])
        data.extend([jax.tree_map(lambda a: a[i], sample) for i in range(size)])
    return data


def batched_generator_from_list(
    data: list,
    batchsize: int,
    shuffle: bool = True,
    drop_last: bool = True,
    seed: int = 1,
    output_transform: Optional[Callable[[jax.Array, PyTree], PyTree]] = None,
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
        batch = tree_batch(data[start:stop], backend="jax")
        batch = batch if output_transform is None else output_transform(key, batch)

        i = (i + 1) % N
        return batch

    return generator


def _data_fn_from_paths(
    paths: list[str],
    include_samples: Optional[list[int]] = None,
):
    "`data_fn` returns numpy arrays."
    # expanduser
    paths = [utils.parse_path(p, mkdir=False) for p in paths]

    extensions = list(set([Path(p).suffix for p in paths]))
    assert len(extensions) == 1

    if extensions[0] == ".h5":
        N = sum([utils.hdf5_load_length(p) for p in paths])

        def data_fn(indices: list[int]):
            return utils.hdf5_load_from_multiple(paths, indices)

    else:
        # TODO
        from x_xy.subpkgs import ml

        list_of_data = []
        for p in paths:
            list_of_data += ml.load(p)

        N = len(list_of_data)

        def data_fn(indices: list[int]):
            return tree_batch([list_of_data[i] for i in indices], backend="numpy")

    if include_samples is None:
        include_samples = list(range(N))
    else:
        # safety copy; we shuffle it in the next function
        include_samples = include_samples.copy()

    return data_fn, include_samples


def _to_jax(tree):
    return jax.tree_map(jnp.asarray, tree)


def _generator_from_data_fn_torch(
    data_fn,
    include_samples: list[int],
    output_transform,
    shuffle: bool,
    batchsize: int,
):
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset

    class _Dataset(Dataset):
        def __len__(self):
            return len(include_samples)

        def __getitem__(self, idx: int):
            element = data_fn([include_samples[idx]])
            if output_transform is not None:
                element = output_transform(element)
            return jax.tree_map(lambda a: a[0], element)

    dl = DataLoader(
        _Dataset(), batch_size=batchsize, shuffle=shuffle, collate_fn=tree_batch
    )
    dl_iter = iter(dl)

    def generator(key: jax.Array):
        nonlocal dl, dl_iter
        try:
            return next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            return next(dl_iter)

    return generator


def _generator_from_data_fn_notorch(
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
        return _to_jax(batch)

    return generator


def batched_generator_from_paths(
    paths: list[str],
    batchsize: int,
    include_samples: Optional[list[int]] = None,
    shuffle: bool = True,
    seed: int = 1,
    output_transform: Optional[
        Callable[[PyTree[np.ndarray]], PyTree[np.ndarray]]
    ] = None,
    use_torch: bool = False,
):
    data_fn, include_samples = _data_fn_from_paths(paths, include_samples)

    N = len(include_samples)
    assert N >= batchsize

    random.seed(seed)
    np.random.seed(seed)

    if use_torch:
        generator = _generator_from_data_fn_torch(
            data_fn, include_samples, output_transform, shuffle, batchsize
        )
    else:
        generator = _generator_from_data_fn_notorch(
            data_fn, include_samples, output_transform, shuffle, batchsize
        )

    return generator, N


def batch_generators_eager(
    generators: Generator | list[Generator],
    sizes: int | list[int],
    batchsize: int,
    shuffle: bool = True,
    drop_last: bool = True,
    seed: int = 1,
    transfer_to_cpu: bool = True,
) -> BatchedGenerator:
    """Eagerly create a large precomputed generator by calling multiple generators
    and stacking their output."""

    data = batch_generators_eager_to_list(
        generators, sizes, seed=seed, transfer_to_cpu=transfer_to_cpu
    )
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
