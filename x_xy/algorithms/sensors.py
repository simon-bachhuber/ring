from typing import Optional

import jax
import jax.numpy as jnp

from x_xy import base, maths, scan


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
    rotvec = maths.quat_to_rotvec(dq)
    gyr = rotvec / dt
    angle = maths.safe_norm(rotvec)
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
        key, c1, c2 = jax.random.split(key, 3)
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
    smoothen_degree: Optional[int] = None,
) -> dict:
    "Simulates a 6D IMU."
    measurements = {"acc": accelerometer(xs, gravity, dt), "gyr": gyroscope(xs.rot, dt)}

    if smoothen_degree is not None:
        measurements = jax.tree_map(
            lambda arr: moving_average(arr, smoothen_degree), measurements
        )

    if noisy:
        assert key is not None, "For noisy sensors random seed `key` must be provided."
        measurements = add_noise_bias(key, measurements)

    return measurements


def rel_pose(
    sys_scan: base.System, xs: base.Transform, sys_xs: Optional[base.System] = None
) -> dict:
    """Relative pose of the entire system. `sys_scan` defines the parent-child ordering,
    relative pose is from child to parent in local coordinates. Bodies that connect
    to the base are skipped (that would be absolute pose).

    Args:
        sys_scan (base.System): System defining parent-child ordering.
        xs (base.Transform): Body transforms from base to body.
        sys_xs (base.System): System that defines the stacking order of `xs`.

    Returns:
        dict:
    """
    if sys_xs is None:
        sys_xs = sys_scan

    if xs.pos.ndim == 3:
        # swap (n_timesteps, n_links) axes
        xs = xs.transpose([1, 0, 2])

    assert xs.batch_dim() == sys_xs.num_links()

    qrel = lambda q1, q2: maths.quat_mul(q1, maths.quat_inv(q2))

    y = {}

    def pose_child_to_parent(_, __, name_i: str, p: int):
        # body connects to base
        if p == -1:
            return

        name_p = sys_scan.idx_to_name(p)

        # find the transforms of those named bodies
        i = sys_xs.name_to_idx(name_i)
        p = sys_xs.name_to_idx(name_p)

        # get those transforms
        q1, q2 = xs.take(p).rot, xs.take(i).rot

        y[name_i] = qrel(q1, q2)

    scan.tree(
        sys_scan, pose_child_to_parent, "ll", sys_scan.link_names, sys_scan.link_parents
    )

    return y


def moving_average(arr, window: int = 7):
    assert window % 2 == 1
    assert window > 1, "Window size of 1 would be a no-op"
    arr_smooth = jnp.zeros((len(arr) + window - 1,) + arr.shape[1:])
    half_window = (window - 1) // 2
    arr_padded = arr_smooth.at[half_window : (len(arr) + half_window)].set(arr)
    arr_padded = arr_padded.at[:half_window].set(arr[0])
    arr_padded = arr_padded.at[-half_window:].set(arr[-1])

    for i in range(-half_window, half_window + 1):
        rolled = jnp.roll(arr_padded, i, axis=0)
        arr_smooth += rolled
    arr_smooth = arr_smooth / window
    return arr_smooth[half_window : (len(arr) + half_window)]
