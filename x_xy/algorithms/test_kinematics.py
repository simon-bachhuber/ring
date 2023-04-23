import jax
import jax.numpy as jnp
import tree_utils as tu

import x_xy
from x_xy import maths


def test_forward_kinematics_transforms():
    sys = x_xy.io.load_example("branched")
    q = [
        jnp.array([1, 0, 0, 0, 1, 1, 1.0]),
        jnp.pi / 2,
        jnp.pi / 2,
        jnp.pi / 4,
        jnp.pi / 2,
    ]
    q = list(map(jnp.atleast_1d, q))
    q = jnp.concatenate(q)
    ts, sys = jax.jit(x_xy.algorithms.forward_kinematics_transforms)(sys, q)

    # position ok
    assert tu.tree_close(ts.take(4).pos, jnp.array([2.0, 2, 1]))

    # orientation ok
    q2_eps = sys.links.transform2.take(2).rot
    q3_2 = sys.links.transform2.take(3).rot
    q4_2 = sys.links.transform.take(4).rot
    assert tu.tree_close(maths.quat_mul(q3_2, q2_eps), ts.take(3).rot)
    assert tu.tree_close(maths.quat_mul(q4_2, q2_eps), ts.take(4).rot)
