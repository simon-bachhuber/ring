from functools import partial

import jax
from jax import custom_jvp
import jax.numpy as jnp
import jax.random as jrand


def wrap_to_pi(phi):
    "Wraps angle `phi` (radians) to interval [-pi, pi]."
    return (phi + jnp.pi) % (2 * jnp.pi) - jnp.pi


x_unit_vector = jnp.array([1.0, 0, 0])
y_unit_vector = jnp.array([0.0, 1, 0])
z_unit_vector = jnp.array([0.0, 0, 1])


def unit_vectors(xyz: int | str):
    if isinstance(xyz, str):
        xyz = {"x": 0, "y": 1, "z": 2}[xyz]
    return [x_unit_vector, y_unit_vector, z_unit_vector][xyz]


@partial(jnp.vectorize, signature="(k)->(1)")
def safe_norm(x):
    """Grad-safe for x=0.0. Norm along last axis."""
    assert x.ndim == 1

    is_zero = jnp.all(jnp.isclose(x, 0.0), axis=-1, keepdims=False)
    return jax.lax.cond(
        is_zero,
        lambda x: jnp.array([0.0], dtype=x.dtype),
        lambda x: jnp.linalg.norm(x, keepdims=True),
        x,
    )


@partial(jnp.vectorize, signature="(k)->(k)")
def safe_normalize(x):
    """Execution- and Grad-safe for x=0.0. Normalizes along last axis."""
    assert x.ndim == 1

    is_zero = jnp.allclose(x, 0.0)
    return jax.lax.cond(
        is_zero,
        lambda x: jnp.zeros_like(x),
        lambda x: x / jnp.where(is_zero, 1.0, safe_norm(x)),
        x,
    )


@custom_jvp
def safe_arccos(x: jnp.ndarray) -> jnp.ndarray:
    """Trigonometric inverse cosine, element-wise with safety clipping in grad."""
    return jnp.arccos(x)


@safe_arccos.defjvp
def _safe_arccos_jvp(primal, tangent):
    (x,) = primal
    (x_dot,) = tangent
    primal_out = safe_arccos(x)
    tangent_out = -x_dot / jnp.sqrt(1.0 - jnp.clip(x, -1 + 1e-7, 1 - 1e-7) ** 2.0)
    return primal_out, tangent_out


@custom_jvp
def safe_arcsin(x: jnp.ndarray) -> jnp.ndarray:
    """Trigonometric inverse sine, element-wise with safety clipping in grad."""
    return jnp.arcsin(x)


@safe_arcsin.defjvp
def _safe_arcsin_jvp(primal, tangent):
    (x,) = primal
    (x_dot,) = tangent
    primal_out = safe_arccos(x)
    tangent_out = x_dot / jnp.sqrt(1.0 - jnp.clip(x, -1 + 1e-7, 1 - 1e-7) ** 2.0)
    return primal_out, tangent_out


@partial(jnp.vectorize, signature="(4)->(4)")
def ensure_positive_w(q):
    return jnp.where(q[0] < 0, -q, q)


def angle_error(q, qhat):
    "Absolute angle in radians between `q` and `qhat`."
    return jnp.abs(quat_angle(quat_mul(quat_inv(q), qhat)))


def inclination_loss(q, qhat):
    """Absolute inclination angle in radians. `q`'s are from body-to-eps.
    This function fullfills
        inclination_loss(q1, q2)
        == inclination_loss(qmt.addHeading(q1, H), q2)
        == inclination_loss(q1, qmt.addHeading(q2, H))`
    for any q1, q2, H
    """
    q_rel = quat_mul(q, quat_inv(qhat))
    q_rel_incl = quat_project(q_rel, [0, 0, 1.0])[1]
    return jnp.abs(quat_angle(q_rel_incl))


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
    """Rotates a vector `vector` by a *unit* quaternion `quat`. q x vec x q^*"""
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
    assert key.shape == (2,), f"{key.shape}"
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


def quat_angle_constantAxisOverTime(qs):
    assert qs.ndim == 2
    assert qs.shape[-1] == 4

    l2norm = lambda x: jnp.sqrt(jnp.sum(x**2, axis=-1))

    axis = safe_normalize(qs[:, 1:])
    angle = quat_angle(qs)[:, None]
    convention = axis[0]
    cond = (l2norm(convention - axis) > l2norm(convention + axis))[..., None]
    return jnp.where(cond, -angle, angle)[:, 0]


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


@partial(jnp.vectorize, signature="(4),(3)->(4),(4)")
def quat_project(q: jax.Array, k: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Decompose quaternion into a primary rotation around axis `k` such that
    the residual rotation's angle is minimized.

    Args:
        q (jax.Array): Quaternion to decompose.
        k (jax.Array): Primary axis direction.

    Returns:
        tuple(jax.Array, jax.Array): Primary quaternion, residual quaternion
    """
    phi_pri = 2 * jnp.arctan2(q[1:] @ k, q[0])
    # NOTE: CONVENTION
    q_pri = quat_rot_axis(k, -phi_pri)
    q_res = quat_mul(q, quat_inv(q_pri))
    return q_pri, q_res


def quat_avg(qs: jax.Array):
    "Tolga Birdal's algorithm."
    if qs.ndim == 1:
        qs = qs[None, :]
    assert qs.ndim == 2
    return jnp.linalg.eigh(
        jnp.einsum("ij,ik,i->...jk", qs, qs, jnp.ones((qs.shape[0],)))
    )[1][:, -1]


# cutoff_freq=20.0; sampe_freq=100.0
# -> alpha = 0.55686
# cutoff_freq=15.0
# -> alpha = 0.48519
def quat_lowpassfilter(
    qs: jax.Array,
    cutoff_freq: float = 20.0,
    samp_freq: float = 100.0,
    filtfilt: bool = False,
) -> jax.Array:
    assert qs.ndim == 2
    assert qs.shape[1] == 4

    if filtfilt:
        qs = quat_lowpassfilter(qs, cutoff_freq, samp_freq, filtfilt=False)
        qs = quat_lowpassfilter(jnp.flip(qs, 0), cutoff_freq, samp_freq, filtfilt=False)
        return jnp.flip(qs, 0)

    omega_times_Ts = 2 * jnp.pi * cutoff_freq / samp_freq
    alpha = omega_times_Ts / (1 + omega_times_Ts)

    def f(y, x):
        # error quaternion; current state -> target
        q_err = quat_mul(x, quat_inv(y))
        # scale down error quaternion
        axis, angle = quat_to_rot_axis(q_err)
        # ensure angle >= 0
        axis, angle = jax.lax.cond(
            angle < 0,
            lambda axis, angle: (-axis, -angle),
            lambda axis, angle: (axis, angle),
            axis,
            angle,
        )
        angle_scaled = angle * alpha
        q_err_scaled = quat_rot_axis(axis, angle_scaled)
        # move small step toward error quaternion
        y = quat_mul(q_err_scaled, y)
        return y, y

    qs_filtered = jax.lax.scan(f, qs[0], qs[1:])[1]

    # padd with first value, such that length remains equal
    qs_filtered = jnp.vstack((qs[0:1], qs_filtered))

    # renormalize due to float32 numerical errors accumulating
    return qs_filtered / jnp.linalg.norm(qs_filtered, axis=-1, keepdims=True)


def quat_inclinationAngle(q: jax.Array):
    head, incl = quat_project(q, jnp.array([0.0, 0, 1]))
    return quat_angle(incl)


def quat_headingAngle(q: jax.Array):
    head, incl = quat_project(q, jnp.array([0.0, 0, 1]))
    return quat_angle(head)


def quat_transfer_heading(q_from: jax.Array, q_to: jax.Array):
    heading = quat_project(q_from, jnp.array([0.0, 0, 1]))[0]
    # set heading to zero in the `q_to` quaternions
    q_to = quat_project(q_to, jnp.array([0.0, 0, 1]))[1]
    return quat_mul(q_to, heading)
