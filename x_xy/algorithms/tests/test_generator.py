import jax
import jax.numpy as jnp
import tree_utils

import x_xy


def finalize_fn_full_imu_setup(key, q, x, sys):
    X = {
        name: x_xy.imu(x.take(sys.name_to_idx(name), 1), sys.gravity, sys.dt)
        for name in sys.link_names
    }
    return X, None


def test_normalize():
    sys = x_xy.load_example("test_three_seg_seg2")
    gen = x_xy.build_generator(sys, finalize_fn=finalize_fn_full_imu_setup)
    gen = x_xy.batch_generator(gen, 50)

    normalizer = x_xy.make_normalizer_from_generator(
        gen, approx_with_large_batchsize=50
    )
    X, _ = gen(jax.random.split(jax.random.PRNGKey(777))[1])
    X = normalizer(X)
    X_flat = tree_utils.batch_concat(X, 2)
    X_mean = jnp.mean(X_flat, (0, 1))
    X_std = jnp.std(X_flat, (0, 1))

    delta = 0.0001
    assert jnp.all(jnp.logical_and(X_mean > -delta, X_mean < delta))
    assert jnp.all(jnp.logical_and(X_std > (1 - delta), X_std < (1 + delta)))
