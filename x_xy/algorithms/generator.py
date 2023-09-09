import math
import random
from typing import Callable

import jax
import jax.numpy as jnp
from tqdm import tqdm
import tree_utils
from tree_utils import PyTree
from tree_utils import tree_batch

from x_xy import utils

from .. import base
from ..scan import scan_sys
from .jcalc import _joint_types
from .jcalc import RCMG_Config
from .kinematics import forward_kinematics_transforms

Generator = Callable[[jax.random.PRNGKey], PyTree]
OfflineGenerator = Callable[[jax.random.PRNGKey], PyTree]
SETUP_FN = Callable[[jax.random.PRNGKey, base.System], base.System]
FINALIZE_FN = Callable[[jax.Array, jax.Array, base.Transform, base.System], PyTree]
Normalizer = Callable[[PyTree], PyTree]


def build_generator(
    sys: base.System,
    config: RCMG_Config = RCMG_Config(),
    setup_fn: SETUP_FN = lambda key, sys: sys,
    finalize_fn: FINALIZE_FN = lambda key, q, x, sys: (q, x),
    randomize_positions: bool = False,
) -> Generator:
    if config.cor:
        for i, p in enumerate(sys.link_parents):
            link_type = sys.link_types[i]
            if p == -1:
                assert link_type == "free"
            if link_type == "free":
                assert p == -1
        sys = sys.replace(
            link_types=["cor" if typ == "free" else typ for typ in sys.link_types]
        )

    def generator(key: jax.random.PRNGKey) -> PyTree:
        nonlocal sys

        # modify system - use `pos_min/max` from xml
        if randomize_positions:
            key, consume = jax.random.split(key)
            sys_mod1 = _setup_fn_randomize_positions(consume, sys)
        else:
            sys_mod1 = sys

        # modify system - use custom logic
        key_start, consume = jax.random.split(key)
        sys_mod2 = setup_fn(consume, sys_mod1)

        # build generalized coordintes vector `q`
        q_list = []

        def draw_q(key, __, link_type, joint_params):
            if key is None:
                key = key_start
            key, key_t, key_value = jax.random.split(key, 3)
            draw_fn = _joint_types[link_type].rcmg_draw_fn
            if draw_fn is None:
                raise Exception(f"The joint type {link_type} has no draw fn specified.")
            q_link = draw_fn(config, key_t, key_value, joint_params)
            # even revolute and prismatic joints must be 2d arrays
            q_link = q_link if q_link.ndim == 2 else q_link[:, None]
            q_list.append(q_link)
            return key

        keys = scan_sys(sys_mod2, draw_q, "ll", sys.link_types, sys.links.joint_params)
        # stack of keys; only the last key is unused
        key = keys[-1]

        q = jnp.concatenate(q_list, axis=1)

        # do forward kinematics
        x, _ = jax.vmap(forward_kinematics_transforms, (None, 0))(sys_mod2, q)

        return finalize_fn(key, q, x, sys_mod2)

    return generator


def _build_batch_matrix(batchsizes: list[int]) -> jax.Array:
    arr = []
    for i, l in enumerate(batchsizes):
        arr += [i] * l
    return jnp.array(arr)


def batch_generator(
    generators: Generator | list[Generator],
    batchsizes: int | list[int] = 1,
    stochastic: bool = False,
) -> Generator:
    """Create a large generator by stacking multiple generators lazily.
    NOTE: If `stochastic` then `batchsizes` must be a single integer.
    """
    if not isinstance(generators, list):
        # test if generator is already batched, then this is a no-op
        key = jax.random.PRNGKey(0)
        X, y = generators(key)
        if tree_utils.tree_ndim(X) > 2:
            return generators

    generators = utils.to_list(generators)

    if stochastic:
        assert isinstance(batchsizes, int)
        bs_total = batchsizes
        pmap, vmap = utils.distribute_batchsize(bs_total)
    else:
        batchsizes = utils.to_list(batchsizes)
        assert len(generators) == len(batchsizes)

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


def offline_generator(
    generators: Generator | list[Generator],
    sizes: int | list[int],
    batchsize: int,
    shuffle: bool = True,
    drop_last: bool = True,
    seed: int = 1,
    store_on_cpu: bool = True,
) -> OfflineGenerator:
    """Eagerly create a large precomputed generator by calling multiple generators
    and stacking their output."""
    assert drop_last, "Not `drop_last` is currently not implemented."
    generators, sizes = utils.to_list(generators), utils.to_list(sizes)
    assert len(generators) == len(sizes)

    key = jax.random.PRNGKey(seed)
    data = []
    for gen, size in tqdm(zip(generators, sizes), desc="offline generator"):
        key, consume = jax.random.split(key)
        sample = batch_generator(gen, size)(consume)
        if store_on_cpu:
            sample = jax.device_put(sample, jax.devices("cpu")[0])
        data.extend([jax.tree_map(lambda a: a[i], sample) for i in range(size)])

    N, i = len(data) // batchsize, 0
    random.seed(seed)

    def generator(key: jax.Array):
        nonlocal i
        del key
        if shuffle and i == 0:
            random.shuffle(data)

        start, stop = i * batchsize, (i + 1) * batchsize
        batch = tree_batch(data[start:stop], backend="jax")
        i = (i + 1) % N
        return batch

    return generator


KEY = jax.random.PRNGKey(777)
KEY_PERMUTATION = jax.random.PRNGKey(888)


def make_normalizer_from_generator(
    generator: Generator, approx_with_large_batchsize: int = 512
) -> Normalizer:
    """`generator` is expected to return `X, y`. Then, this function returns a pure
    function that normalizes `X`."""
    # batch it if it isn't already
    generator = batch_generator(generator)

    # probe generator for its batchsize
    X, _ = generator(KEY)
    bs = tree_utils.tree_shape(X)

    # how often do we have to query the generator
    number_of_gen_calls = math.ceil(approx_with_large_batchsize / bs)

    Xs, key = [], KEY
    for _ in range(number_of_gen_calls):
        key, consume = jax.random.split(key)
        Xs.append(generator(consume)[0])
    Xs = tree_utils.tree_batch(Xs, True, "jax")
    # permute 0-th axis, since batchsize of generator might be larger than
    # `approx_with_large_batchsize`, then we would not get a representative
    # subsample otherwise
    Xs = jax.tree_map(lambda arr: jax.random.permutation(KEY_PERMUTATION, arr), Xs)
    Xs = tree_utils.tree_slice(Xs, start=0, slice_size=approx_with_large_batchsize)

    # obtain statistics
    mean = jax.tree_map(lambda arr: jnp.mean(arr, axis=(0, 1)), Xs)
    std = jax.tree_map(lambda arr: jnp.std(arr, axis=(0, 1)), Xs)

    eps = 1e-8

    def normalizer(X):
        return jax.tree_map(lambda a, b, c: (a - b) / (c + eps), X, mean, std)

    return normalizer


def _setup_fn_randomize_positions(key: jax.Array, sys: base.System) -> base.System:
    ts = sys.links.transform1

    for i in range(sys.num_links()):
        link = sys.links[i]
        key, new_pos = _draw_pos_uniform(key, link.pos_min, link.pos_max)
        ts = ts.index_set(i, ts[i].replace(pos=new_pos))

    return sys.replace(links=sys.links.replace(transform1=ts))


def _draw_pos_uniform(key, pos_min, pos_max):
    key, c1, c2, c3 = jax.random.split(key, num=4)
    pos = jnp.array(
        [
            jax.random.uniform(c1, minval=pos_min[0], maxval=pos_max[0]),
            jax.random.uniform(c2, minval=pos_min[1], maxval=pos_max[1]),
            jax.random.uniform(c3, minval=pos_min[2], maxval=pos_max[2]),
        ]
    )
    return key, pos
