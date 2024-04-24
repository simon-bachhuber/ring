from _compat import unbatch_gen
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ring
from ring.algorithms.jcalc import _find_interval


@pytest.mark.parametrize("T,seed", [(10.0, 0), (20.0, 0), (10.0, 1), (20.0, 1)])
def test_join_motionconfigs(T, seed):
    motion_config = ring.MotionConfig()
    nomotion_config = motion_config.to_nomotion_config()

    sys = ring.io.load_example("test_free")
    dt = sys.dt
    q, x = unbatch_gen(
        ring.RCMG(
            sys,
            ring.join_motionconfigs(
                [nomotion_config, motion_config, nomotion_config], [T, 2 * T]
            ),
            finalize_fn=lambda key, q, x, sys: (q, x),
        ).to_lazy_gen()
    )(jax.random.PRNGKey(seed))

    def array_eq(a: int, b: int):
        "Test if `q` is constant from between `a` and `b` indices."
        np.testing.assert_allclose(
            np.repeat(q[a : (a + 1), :4], b - a, axis=0), q[a:b, :4]
        )
        np.testing.assert_allclose(
            np.repeat(q[a : (a + 1), 4:], b - a, axis=0), q[a:b, 4:]
        )

    T_i = int(T / dt)
    array_eq(0, T_i)

    with pytest.raises(AssertionError):
        T_i += 10
        array_eq(0, T_i)

    T_i = int((2 * T + motion_config.t_max) / dt)
    T_f = int(nomotion_config.T / dt)
    array_eq(T_i, T_f)

    with pytest.raises(AssertionError):
        T_i = int((2 * T) / dt)
        array_eq(T_i, T_f)

    # test that two configs that disagree with values that are not
    # time-dependent can not be concatenated
    motion_config_disagree = ring.MotionConfig(t_min=0.04)
    with pytest.raises(AssertionError):
        ring.join_motionconfigs([motion_config, motion_config_disagree], [T])


def test_find_interval():
    assert _find_interval(1.5, jnp.array([0.0, 1.0, 2.0])) == 2
    assert _find_interval(0.5, jnp.array([0.0])) == 1
    assert _find_interval(-0.5, jnp.array([0.0])) == 0


def test_motion_config_register():
    ring.MotionConfig.from_register("expSlow")
