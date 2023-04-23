import jax
import tree_utils

from x_xy import maths


def test_quat_rot_axis_are_inverse():
    q = maths.quat_random(
        jax.random.PRNGKey(
            1,
        )
    )
    assert tree_utils.tree_close(q, maths.quat_rot_axis(*maths.quat_to_rot_axis(q)))
