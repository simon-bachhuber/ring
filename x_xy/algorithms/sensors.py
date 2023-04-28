from typing import Optional

import jax
import jax.numpy as jnp

from x_xy import base, maths


def accelerometer(xs: base.Transform, gravity: jax.Array, dt: float) -> jax.Array:
    """Compute measurements of an accelerometer that follows a frame which moves along
    a trajectory of Transforms. Let `xs` be the transforms from base to link.
    """

    acc = (xs.pos[:-2] + xs.pos[2:] - 2 * xs.pos[1:-1]) / dt**2
    acc = acc + gravity

    # 2nd order derivative, (N,) -> (N-2,)
    # prepend and append one element to keep shape size
    acc = jnp.vstack((acc[0:1], acc, acc[-1][None]))

    return maths.rotate(acc, xs.rot)


def gyroscope(rot: jax.Array, dt: float) -> jax.Array:
    """Compute measurements of a gyroscope that follows a frame with an orientation
    given by trajectory of quaternions `rot`."""
    # this was not needed before
    # the reason is that before q represented
    # FROM LOCAL TO EPS
    # but now we q represents
    # FROM EPS TO LOCAL
    # q = maths.quat_inv(rot)

    q = rot
    # 1st-order approx to derivative
    dq = maths.quat_mul(q[1:], maths.quat_inv(q[:-1]))

    # due to 1st order derivative, shape (N,) -> (N-1,)
    # append one element at the end to keep shape size
    dq = jnp.vstack((dq, dq[-1][None]))

    axis, angle = maths.quat_to_rot_axis(dq)
    angle = angle[:, None]

    gyr = axis * angle / dt
    return jnp.where(jnp.abs(angle) > 1e-10, gyr, jnp.zeros(3))


NOISE_LEVELS = {"acc": 0.5, "gyr": jnp.deg2rad(1.0)}
BIAS_LEVELS = {"acc": 0.5, "gyr": jnp.deg2rad(1.0)}


def add_noise_bias(key: jax.random.PRNGKey, imu_measurements: dict) -> dict:
    """Add noise and bias to 6D imu measurements.

    Args:
        key (jax.random.PRNGKey): Random seed.
        imu_measurements (dict): IMU measurements without noise and bias.
            Format is {"gyr": Array, "acc": Array}.

    Returns:
        dict: IMU measurements with noise and bias.
    """
    noisy_imu_measurements = {}
    for sensor in ["acc", "gyr"]:
        key, c1, c2 = jax.random.split(key)
        noise = (
            jax.random.normal(c1, shape=imu_measurements[sensor].shape)
            * NOISE_LEVELS[sensor]
        )
        bias = jax.random.uniform(
            c2, minval=-BIAS_LEVELS[sensor], maxval=BIAS_LEVELS[sensor]
        )
        noisy_imu_measurements[sensor] = imu_measurements[sensor] + noise + bias
    return noisy_imu_measurements


def imu(
    xs: base.Transform,
    gravity: jax.Array,
    dt: float,
    key: Optional[jax.random.PRNGKey] = None,
    noisy: bool = False,
):
    "Simulates a 6D IMU."
    measurements = {"acc": accelerometer(xs, gravity, dt), "gyr": gyroscope(xs.rot, dt)}

    if noisy:
        assert key is not None, "For noisy sensors random seed `key` must be provided."
        measurements = add_noise_bias(key, measurements)

    return measurements
