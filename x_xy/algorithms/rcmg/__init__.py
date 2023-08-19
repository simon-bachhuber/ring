import math
from typing import Callable

import jax
import jax.numpy as jnp
import tree_utils
from tree_utils import PyTree

from x_xy import base, scan, utils
from x_xy.algorithms import RCMG_Config, forward_kinematics_transforms
from x_xy.algorithms.jcalc import _joint_types
from x_xy.algorithms.rcmg.augmentations import (
    register_rr_joint,
    replace_free_with_cor,
    setup_fn_randomize_joint_axes,
    setup_fn_randomize_positions,
)

Generator = Callable[[jax.random.PRNGKey], PyTree]
SETUP_FN = Callable[[jax.random.PRNGKey, base.System], base.System]
FINALIZE_FN = Callable[[jax.Array, jax.Array, base.Transform, base.System], PyTree]
Normalizer = Callable[[PyTree], PyTree]


def build_generator(
    sys: base.System,
    config: RCMG_Config = RCMG_Config(),
    setup_fn: SETUP_FN = lambda key, sys: sys,
    finalize_fn: FINALIZE_FN = lambda key, q, x, sys: (q, x),
) -> Generator:
    def generator(key: jax.random.PRNGKey) -> PyTree:
        nonlocal sys
        # modified system
        key_start, consume = jax.random.split(key)
        sys_mod = setup_fn(consume, sys)

        # build generalized coordintes vector `q`
        q_list = []

        def draw_q(key, __, link_type):
            if key is None:
                key = key_start
            key, key_t, key_value = jax.random.split(key, 3)
            draw_fn = _joint_types[link_type].rcmg_draw_fn
            if draw_fn is None:
                raise Exception(f"The joint type {link_type} has no draw fn specified.")
            q_link = draw_fn(config, key_t, key_value)
            # even revolute and prismatic joints must be 2d arrays
            q_link = q_link if q_link.ndim == 2 else q_link[:, None]
            q_list.append(q_link)
            return key

        keys = scan.tree(sys_mod, draw_q, "l", sys.link_types)
        # stack of keys; only the last key is unused
        key = keys[-1]

        q = jnp.concatenate(q_list, axis=1)

        # do forward kinematics
        x, _ = jax.vmap(forward_kinematics_transforms, (None, 0))(sys_mod, q)

        return finalize_fn(key, q, x, sys_mod)

    return generator


def _build_batch_matrix(batchsizes: list[int]) -> jax.Array:
    arr = []
    for i, l in enumerate(batchsizes):
        arr += [i] * l
    return jnp.array(arr)


def batch_generator(
    generators: Generator | list[Generator], batchsizes: int | list[int] = 1
) -> Generator:
    if not isinstance(generators, list):
        # test if generator is already batched, then this is a no-op
        key = jax.random.PRNGKey(0)
        X, y = generators(key)
        if tree_utils.tree_ndim(X) > 2:
            return generators

    if not isinstance(generators, list):
        generators = [generators]
    if not isinstance(batchsizes, list):
        batchsizes = [batchsizes]

    assert len(generators) == len(batchsizes)

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
