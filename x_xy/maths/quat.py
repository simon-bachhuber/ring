from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrand

from .basic import wrap_to_pi
from .safe import safe_arcsin
from .safe import safe_norm
from .safe import safe_normalize


@partial(jnp.vectorize, signature="(4)->(4)")
def ensure_positive_w(q):
    return jnp.where(q[0] < 0, -q, q)


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
    NOTE: Usually, we actually want the second interpretation. Think about it,
    we use quaternions to re-express vectors in other frames. But the
    vectors stay the same. We only transform them to a common frames.
    """
    assert axis.shape == (3,)
    assert angle.shape == ()

    axis = safe_normalize(axis)
    # NOTE: CONVENTION
    # 23.04.23
    # this fixes the issue of prismatic joints being inverted w.r.t.
    # gravity vector.
    # The reason is that it inverts the way how revolute joints behave
    # Such that prismatic joints work by inverting gravity
    angle *= -1.0
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


def quat_random(
    key: jrand.PRNGKey, batch_shape: tuple = (), maxval: float = jnp.pi
) -> jax.Array:
    """Provides a random *unit* quaternion, sampled uniformly"""
    shape = batch_shape + (4,)
    qs = safe_normalize(jrand.normal(key, shape))

    def _scale_angle():
        axis, angle = quat_to_rot_axis(qs)
        angle_scaled = angle * maxval / jnp.pi
        return quat_rot_axis(axis, angle_scaled)

    return jax.lax.cond(maxval == jnp.pi, lambda: qs, _scale_angle)


def quat_euler(angles, intrinsic=True, convention="zyx"):
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
            return quat_mul(q3, quat_mul(q2, q1))
        else:
            return quat_mul(q1, quat_mul(q2, q3))

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
    # NOTE: CONVENTION
    angle *= -1.0
    axis = safe_normalize(q[1:])
    return axis, angle


@partial(jnp.vectorize, signature="(3)->(4)")
def euler_to_quat(angles: jnp.ndarray) -> jnp.ndarray:
    """Converts euler rotations in radians to quaternion."""
    # this follows the Tait-Bryan intrinsic rotation formalism: x-y'-z''
    c1, c2, c3 = jnp.cos(angles / 2)
    s1, s2, s3 = jnp.sin(angles / 2)
    w = c1 * c2 * c3 - s1 * s2 * s3
    x = s1 * c2 * c3 + c1 * s2 * s3
    y = c1 * s2 * c3 - s1 * c2 * s3
    z = c1 * c2 * s3 + s1 * s2 * c3
    # NOTE: CONVENTION
    return quat_inv(jnp.array([w, x, y, z]))


@partial(jnp.vectorize, signature="(4)->(3)")
def quat_to_euler(q: jnp.ndarray) -> jnp.ndarray:
    """Converts quaternions to euler rotations in radians."""
    # this follows the Tait-Bryan intrinsic rotation formalism: x-y'-z''

    # NOTE: CONVENTION
    q = quat_inv(q)

    z = jnp.arctan2(
        -2 * q[1] * q[2] + 2 * q[0] * q[3],
        q[1] * q[1] + q[0] * q[0] - q[3] * q[3] - q[2] * q[2],
    )
    # TODO: Investigate why quaternions go so big we need to clip.
    y = safe_arcsin(jnp.clip(2 * q[1] * q[3] + 2 * q[0] * q[2], -1.0, 1.0))
    x = jnp.arctan2(
        -2 * q[2] * q[3] + 2 * q[0] * q[1],
        q[3] * q[3] - q[2] * q[2] - q[1] * q[1] + q[0] * q[0],
    )

    return jnp.array([x, y, z])
