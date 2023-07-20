import jax
import jax.numpy as jnp
import tree_utils

import x_xy
from x_xy.algorithms.rcmg import make_normalizer_from_generator


def test_normalize():
    sys = x_xy.io.load_example("test_three_seg_seg2")
    gen = x_xy.algorithms.build_generator(sys)
    gen = x_xy.algorithms.batch_generator(gen, 50)

    normalizer = make_normalizer_from_generator(gen, approx_with_large_batchsize=50)
    X, _ = gen(jax.random.split(jax.random.PRNGKey(777))[1])
    X = normalizer(X)
    X_flat = tree_utils.batch_concat(X, 2)
    X_mean = jnp.mean(X_flat, (0, 1))
    X_std = jnp.std(X_flat, (0, 1))

    delta = 0.0001
    assert jnp.all(jnp.logical_and(X_mean > -delta, X_mean < delta))
    assert jnp.all(jnp.logical_and(X_std > (1 - delta), X_std < (1 + delta)))
