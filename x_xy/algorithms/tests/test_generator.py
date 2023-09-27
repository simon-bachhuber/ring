import jax
import jax.numpy as jnp
import numpy as np
import tree_utils

import x_xy
from x_xy.algorithms.generator.transforms import _draw_pos_uniform
from x_xy.algorithms.generator.transforms import _setup_fn_randomize_positions
from x_xy.subpkgs import sys_composer


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


def setup_fn_old(key, sys: x_xy.System) -> x_xy.System:
    def replace_pos(transforms, new_pos, name: str):
        i = sys.name_to_idx(name)
        return transforms.index_set(i, transforms[i].replace(pos=new_pos))

    ts = sys.links.transform1

    # seg 1 relative to seg2
    key, pos = _draw_pos_uniform(key, [-0.2, -0.02, -0.02], [-0.0, 0.02, 0.02])
    ts = replace_pos(ts, pos, "seg1")

    # imu1 relative to seg1
    key, pos = _draw_pos_uniform(key, [-0.25, -0.05, -0.05], [-0.05, 0.05, 0.05])
    ts = replace_pos(ts, pos, "imu1")

    # seg3 relative to seg2
    key, pos = _draw_pos_uniform(key, [0.0, -0.02, -0.02], [0.2, 0.02, 0.02])
    ts = replace_pos(ts, pos, "seg3")

    # seg4 relative to seg3
    key, pos = _draw_pos_uniform(key, [0.0, -0.02, -0.02], [0.4, 0.02, 0.02])
    ts = replace_pos(ts, pos, "seg4")

    # imu2 relative to seg3
    key, pos = _draw_pos_uniform(key, [0.05, -0.05, -0.05], [0.25, 0.05, 0.05])
    ts = replace_pos(ts, pos, "imu2")

    return sys.replace(links=sys.links.replace(transform1=ts))


def test_randomize_positions():
    key = jax.random.PRNGKey(1)
    sys = x_xy.load_example("test_randomize_position")

    # split key once more because the new logic `setup_fn_randomize_positions`
    # randomizes the position for each body even if the body has
    # no explicit `pos_min` and `pos_max` given in the xml
    # this is the case here for the body `seg2`
    # i.e. this is the split for `seg2` relative to `worldbody`
    internal_key, *_ = jax.random.split(key, 4)
    # then comes `seg1` relative to `seg2`
    pos_old = setup_fn_old(internal_key, sys).links.transform1.pos

    pos_new = _setup_fn_randomize_positions(key, sys).links.transform1.pos

    np.testing.assert_array_equal(pos_old, pos_new)


def test_cor():
    sys = x_xy.load_example("test_three_seg_seg2")
    sys = sys_composer.inject_system(sys, sys.add_prefix_suffix("second_"))
    x_xy.build_generator(sys, x_xy.RCMG_Config(cor=True))(jax.random.PRNGKey(1))
