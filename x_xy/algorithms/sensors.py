from typing import Optional

import jax
import jax.numpy as jnp

from x_xy import algebra
from x_xy import maths

from .. import base
from ..io import load_sys_from_str
from ..scan import scan_sys
from .dynamics import step


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
    delay: Optional[int] = None,
    random_s2s_ori: bool = False,
    quasi_physical: bool = False,
) -> dict:
    """Simulates a 6D IMU, `xs` should be Transforms from eps-to-imu.
    NOTE: `smoothen_degree` is used as window size for moving average.
    NOTE: If `smoothen_degree` is given, and `delay` is not, then delay is chosen
    such moving average window is delayed to just be causal.
    """
    assert xs.ndim() == 2

    if random_s2s_ori:
        assert key is not None, "`random_s2s_ori` requires a random seed via `key`"
        # `xs` are now from eps-to-segment, so add another final rotation from
        # segment-to-sensor where this transform is only rotational
        key, consume = jax.random.split(key)
        xs_s2s = base.Transform.create(rot=maths.quat_random(consume))
        xs = jax.vmap(algebra.transform_mul, in_axes=(None, 0))(xs_s2s, xs)

    if quasi_physical:
        xs = _quasi_physical_simulation(xs, dt)

    measurements = {"acc": accelerometer(xs, gravity, dt), "gyr": gyroscope(xs.rot, dt)}

    if smoothen_degree is not None:
        measurements = jax.tree_map(
            lambda arr: moving_average(arr, smoothen_degree),
            measurements,
        )

        # if you low-pass filter the imu measurements through a moving average which
        # effectively uses future values, then it also makes sense to delay the imu
        # measurements by this amount such that no future information is used
        if delay is None:
            half_window = (smoothen_degree - 1) // 2
            delay = half_window

    if delay is not None and delay > 0:
        measurements = jax.tree_map(
            lambda arr: (jnp.pad(arr, ((delay, 0), (0, 0)))[:-delay]), measurements
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
        dict: Child-to-parent quaternions
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

    scan_sys(
        sys_scan, pose_child_to_parent, "ll", sys_scan.link_names, sys_scan.link_parents
    )

    return y


def moving_average(arr: jax.Array, window: int) -> jax.Array:
    "Padds with left and right values of array."
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


_quasi_physical_sys_str = r"""
<x_xy>
    <options gravity="0 0 0"/>
    <worldbody>
        <body name="IMU" joint="p3d" damping="0.1 0.1 0.1" spring_stiff="3 3 3">
            <geom type="box" mass="0.002" dim="0.01 0.01 0.01"/>
        </body>
    </worldbody>
</x_xy>
"""


def _quasi_physical_simulation_beautiful(
    xs: base.Transform, dt: float
) -> base.Transform:
    sys = load_sys_from_str(_quasi_physical_sys_str).replace(dt=dt)

    def step_dynamics(state: base.State, x):
        state = step(sys.replace(link_spring_zeropoint=x.pos), state)
        return state, state.q

    state = base.State.create(sys, q=xs.pos[0])
    _, pos = jax.lax.scan(step_dynamics, state, xs)
    return xs.replace(pos=pos)


_constants = {
    "qp_damp": 7.0,
    "qp_stif": 125.0,
}


def _quasi_physical_simulation(xs: base.Transform, dt: float) -> base.Transform:
    mass = 1.0
    damp = _constants["qp_damp"]
    stif = _constants["qp_stif"]

    def step_dynamics(state, zeropoint):
        pos, vel = state
        zeropoint_pos, zeropoint_vel = zeropoint
        acc = (damp * (zeropoint_vel - vel) + stif * (zeropoint_pos - pos)) / mass
        vel += dt * acc
        # semi-implicit, so use already next velocity
        pos += dt * vel
        return (pos, vel), pos

    zero_vel = jnp.zeros_like(xs.pos[0])
    state = (xs.pos[0], zero_vel)
    zeropoint_vel = jnp.vstack((zero_vel, jnp.diff(xs.pos, axis=0) / dt))
    zeropoint = (xs.pos, zeropoint_vel)
    _, pos = jax.lax.scan(step_dynamics, state, zeropoint)
    return xs.replace(pos=pos)
