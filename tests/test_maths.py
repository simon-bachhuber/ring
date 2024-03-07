import jax
import jax.numpy as jnp
import numpy as np
from ring import maths
import tree_utils


def test_quat_rot_axis_are_inverse():
    q = maths.quat_random(
        jax.random.PRNGKey(
            1,
        )
    )
    axis, angle = maths.quat_to_rot_axis(q)
    assert tree_utils.tree_close(q, maths.quat_rot_axis(axis, angle))


def test_3x3_are_inverse():
    q = maths.quat_random(
        jax.random.PRNGKey(
            1,
        )
    )
    mat = maths.quat_to_3x3(q)

    np.testing.assert_allclose(q, maths.quat_from_3x3(mat), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        mat, maths.quat_to_3x3(maths.quat_from_3x3(mat)), rtol=1e-5, atol=1e-5
    )


def test_quat_convention():
    q = maths.quat_euler([0, jnp.deg2rad(90), 0])
    np.testing.assert_allclose(
        maths.rotate(jnp.array([1.0, 0, 0]), q), jnp.array([0, 0, 1.0]), atol=1e-7
    )

    mat = maths.quat_to_3x3(q)
    np.testing.assert_allclose(
        mat @ jnp.array([1.0, 0, 0]), jnp.array([0, 0, 1.0]), atol=1e-6
    )

    q1 = maths.quat_random(
        jax.random.PRNGKey(
            1,
        )
    )

    q2 = maths.quat_mul(q, q1)
    q = maths.quat_mul(q2, maths.quat_inv(q1))
    axis, angle = maths.quat_to_rot_axis(q)
    # negative axis and angle cancel
    np.testing.assert_allclose(axis, jnp.array([0, -1.0, 0]), atol=1e-7)
    np.testing.assert_allclose(angle, -jnp.deg2rad(90), atol=1e-7)


def test_euler_angles():
    # test that two functions `quat_euler` and `euler_to_quat` are equal
    angles = jnp.array([1.2, 0.55, 0.25])
    np.testing.assert_allclose(
        maths.quat_euler(angles, convention="xyz"), maths.euler_to_quat(angles)
    )

    # test that they are inverses
    np.testing.assert_allclose(
        angles, maths.quat_to_euler(maths.euler_to_quat(angles)), atol=1e-7, rtol=1e-6
    )


def test_quat_project():
    q = maths.quat_random(
        jax.random.PRNGKey(
            1,
        )
    )

    q_pri, q_res = maths.quat_project(q, jnp.array([1.0, 0, 0]))
    np.testing.assert_allclose(q, maths.quat_mul(q_res, q_pri), 1e-6, 1e-7)

    axis, angle = maths.quat_to_rot_axis(q)
    # NOTE: CONVENTION
    angle *= -1
    q_pri, q_res = maths.quat_project(q, axis)
    np.testing.assert_allclose(angle, maths.quat_angle(q_pri))
    np.testing.assert_allclose(jnp.array(0.0), maths.quat_angle(q_res))
