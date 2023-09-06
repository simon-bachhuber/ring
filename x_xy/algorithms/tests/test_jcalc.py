import jax
import jax.numpy as jnp
import numpy as np
import pytest

import x_xy
from x_xy.algorithms.jcalc import _find_interval


@pytest.mark.parametrize("T,seed", [(10.0, 0), (20.0, 0), (10.0, 1), (20.0, 1)])
def test_concat_configs(T, seed):
    nomotion_config = x_xy.RCMG_Config(
        dang_min=0.0,
        dang_max=0.0,
        dang_max_free_spherical=0.0,
        dang_min_free_spherical=0.0,
        dpos_min=0.0,
        dpos_max=0.0,
    )
    motion_config = x_xy.RCMG_Config()

    sys = x_xy.load_example("test_free")
    q, x = x_xy.build_generator(
        sys,
        x_xy.concat_configs(
            [nomotion_config, motion_config, nomotion_config], [T, 2 * T]
        ),
    )(jax.random.PRNGKey(seed))

    def array_eq(a: int, b: int):
        np.testing.assert_allclose(
            np.repeat(q[a : (a + 1), :4], b - a, axis=0), q[a:b, :4]
        )
        np.testing.assert_allclose(
            np.repeat(q[a : (a + 1), 4:], b - a, axis=0), q[a:b, 4:]
        )

    T_i = int(T / nomotion_config.Ts)
    array_eq(0, T_i)

    with pytest.raises(AssertionError):
        T_i += 10
        array_eq(0, T_i)

    T_i = int((2 * T + motion_config.t_max) / nomotion_config.Ts)
    T_f = int(nomotion_config.T / nomotion_config.Ts)
    array_eq(T_i, T_f)

    with pytest.raises(AssertionError):
        T_i -= 10
        array_eq(T_i, T_f)


def test_find_interval():
    assert _find_interval(1.5, jnp.array([0.0, 1.0, 2.0])) == 2
    assert _find_interval(0.5, jnp.array([0.0])) == 1
    assert _find_interval(-0.5, jnp.array([0.0])) == 0
