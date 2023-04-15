from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrand

from .basic import wrap_to_pi
from .safe import safe_norm, safe_normalize


def angle_error(q, qhat):
    "Angle in radians between `q` and `qhat`."
    return jnp.abs(quat_angle(quat_mul(quat_inv(q), qhat)))


def unit_quats_like(array):
    "Array of *unit* quaternions of identical shape."
    if array.shape[-1] != 4:
        raise Exception()

    return jnp.ones(array.shape[:-1])[..., None] * jnp.array([1.0, 0, 0, 0])


@partial(jnp.vectorize, signature="(4),(4)->(4)")
def quat_mul(u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    "Multiplies two quaternions."
    q = jnp.array(
        [
            u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
            u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
            u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
            u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
        ]
    )
    return q


def quat_inv(q: jnp.ndarray) -> jnp.ndarray:
    "Calculates the inverse of quaternion q."
    return q * jnp.array([1.0, -1, -1, -1])


@partial(jnp.vectorize, signature="(3),(4)->(3)")
def rotate(vector: jnp.ndarray, quat: jnp.ndarray):
    """Rotates a vector `vector` by a *unit* quaternion `quat`."""
    qvec = jnp.array([0, *vector])
    return rotate_quat(qvec, quat)[1:4]


def rotate_matrix(matrix: jax.Array, quat: jax.Array):
    "Rotate matrix `matrix` by a *unit* quaternion `quat`."
    E = quat_to_3x3(quat)
    return E @ matrix @ E.T


def rotate_quat(q: jax.Array, quat: jax.Array):
    "Rotate quaternion `q` by `quat`"
    return quat_mul(quat, quat_mul(q, quat_inv(quat)))


@partial(jnp.vectorize, signature="(3),()->(4)")
def quat_rot_axis(axis: jnp.ndarray, angle: jnp.ndarray) -> jnp.ndarray:
    """Construct a *unit* quaternion that describes rotating around
    `axis` by `angle` (radians).

    This is the interpretation of rotating the vector and *not*
    the frame.
    For the interpretation of rotating the frame and *not* the
    vector, you should use angle -> -angle.
    """
    axis = safe_normalize(axis)
    s, c = jnp.sin(angle / 2), jnp.cos(angle / 2)
    return jnp.array([c, *(axis * s)])


@partial(jnp.vectorize, signature="(3,3)->(4)")
def quat_from_3x3(m: jnp.ndarray) -> jnp.ndarray:
    """Converts 3x3 rotation matrix to *unit* quaternion."""
    w = jnp.sqrt(1 + m[0, 0] + m[1, 1] + m[2, 2]) / 2.0
    x = (m[2][1] - m[1][2]) / (w * 4)
    y = (m[0][2] - m[2][0]) / (w * 4)
    z = (m[1][0] - m[0][1]) / (w * 4)
    return jnp.array([w, x, y, z])


@partial(jnp.vectorize, signature="(4)->(3,3)")
def quat_to_3x3(q: jnp.ndarray) -> jnp.ndarray:
    """Converts *unit* quaternion to 3x3 rotation matrix."""
    d = jnp.dot(q, q)
    w, x, y, z = q
    s = 2 / d
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs

    return jnp.array(
        [
            jnp.array([1 - (yy + zz), xy - wz, xz + wy]),
            jnp.array([xy + wz, 1 - (xx + zz), yz - wx]),
            jnp.array([xz - wy, yz + wx, 1 - (xx + yy)]),
        ]
    )


def quat_random(key: jrand.PRNGKey, batch_shape: tuple[int] = ()) -> jax.Array:
    """Provides a random *unit* quaternion, sampled uniformly"""
    shape = batch_shape + (4,)
    return safe_normalize(jrand.normal(key, shape))


def quat_euler(angles, intrinsic=True, convention="xyz"):
    "Construct a *unit* quaternion from Euler angles (radians)."

    @partial(jnp.vectorize, signature="(3)->(4)")
    def _quat_euler(angles):
        xunit = jnp.array([1.0, 0.0, 0.0])
        yunit = jnp.array([0.0, 1.0, 0.0])
        zunit = jnp.array([0.0, 0.0, 1.0])

        axes_map = {
            "x": xunit,
            "y": yunit,
            "z": zunit,
        }

        q1 = quat_rot_axis(axes_map[convention[0]], angles[0])
        q2 = quat_rot_axis(axes_map[convention[1]], angles[1])
        q3 = quat_rot_axis(axes_map[convention[2]], angles[2])

        if intrinsic:
            return quat_mul(q1, quat_mul(q2, q3))
        else:
            return quat_mul(q3, quat_mul(q2, q1))

    return _quat_euler(angles)


@partial(jnp.vectorize, signature="(4)->()")
def quat_angle(q):
    "Extract rotation angle (radians) of quaternion `q`."
    phi = 2 * jnp.arctan2(safe_norm(q[1:])[0], q[0])
    return wrap_to_pi(phi)


@partial(jnp.vectorize, signature="(4)->(3),()")
def quat_to_rot_axis(q):
    "Extract unit-axis and angle from quaternion `q`."
    angle = quat_angle(q)
    axis = safe_normalize(q[1:])
    return axis, angle
