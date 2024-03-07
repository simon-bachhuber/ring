import math
from typing import Callable, TypeVar

import jax
import jax.numpy as jnp
from ring.algorithms.generator import types
import tree_utils

KEY = jax.random.PRNGKey(777)
KEY_PERMUTATION = jax.random.PRNGKey(888)


X = TypeVar("X")
Normalizer = Callable[[X], X]


def make_normalizer_from_generator(
    generator: types.BatchedGenerator,
    approx_with_large_batchsize: int = 512,
    verbose: bool = False,
) -> Normalizer:
    "Returns a pure function that normalizes `X`."

    # probe generator for its batchsize
    X, _ = generator(KEY)
    bs = tree_utils.tree_shape(X)
    assert tree_utils.tree_ndim(X) == 3, "`generator` must be batched."

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

    if verbose:
        print("Mean: ", mean)
        print("Std: ", std)

    eps = 1e-8

    def normalizer(X):
        return jax.tree_map(lambda a, b, c: (a - b) / (c + eps), X, mean, std)

    return normalizer
