from collections import defaultdict
from typing import Optional

import jax
import jax.numpy as jnp

from ring import algebra
from ring import algorithms
from ring import base
from ring import io
from ring import maths
from ring import sim2real


def accelerometer(
    xs: base.Transform, gravity: jax.Array, dt: float, n: int
) -> jax.Array:
    """Compute measurements of an accelerometer that follows a frame which moves along
    a trajectory of Transforms. Let `xs` be the transforms from base to link.
    """

    acc = (xs.pos[: -2 * n] + xs.pos[2 * n :] - 2 * xs.pos[n:-n]) / (n * dt) ** 2
    acc = acc + gravity

    # 2nd order derivative, (N,) -> (N-2n,)
    # prepend and append n elements to keep shape size
    acc = jnp.vstack((jnp.atleast_2d(acc[:n]), acc, jnp.atleast_2d(acc[-n:])))

    return maths.rotate(acc, xs.rot)


def gyroscope(rot: jax.Array, dt: float, second_order: bool) -> jax.Array:
    """Compute measurements of a gyroscope that follows a frame with an orientation
    given by trajectory of quaternions `rot`."""
    # this was not needed before
    # the reason is that before q represented
    # FROM LOCAL TO EPS
    # but now we q represents
    # FROM EPS TO LOCAL
    # q = maths.quat_inv(rot)

    q = rot
    if second_order:
        dq = maths.quat_mul(q[2:], maths.quat_inv(q[:-2]))
        dq = jnp.vstack((dq[0][None], dq, dq[-1][None]))

        dt = 2 * dt
    else:
        # 1st-order approx to derivative
        dq = maths.quat_mul(q[1:], maths.quat_inv(q[:-1]))

        # due to 1st order derivative, shape (N,) -> (N-1,)
        # append one element at the end to keep shape size
        dq = jnp.vstack((dq, dq[-1][None]))

    axis, angle = maths.quat_to_rot_axis(dq)
    angle = angle[:, None]

    gyr = axis * angle / dt
    return jnp.where(jnp.abs(angle) > 1e-10, gyr, jnp.zeros(3))


def _draw_random_magvec(key):
    "Unit is in a.u. (40 microTesla)"
    c1, c2 = jax.random.split(key)

    dip_angle_min, dip_angle_max = -85.0, -50.0  # degrees
    dip_angle = jnp.deg2rad(
        jax.random.uniform(c1, minval=dip_angle_min, maxval=dip_angle_max)
    )

    norm_minval, norm_maxval = 15e-6, 65e-6  # Tesla; from lecture script page 10
    # convert Tesla -> a.u. where (1 a.u. ~ 40 microTesla)
    au = 40e-6
    norm_minval, norm_maxval = norm_minval / au, norm_maxval / au
    norm = jax.random.uniform(c2, minval=norm_minval, maxval=norm_maxval)

    return jnp.array([0.0, jnp.cos(dip_angle), jnp.sin(dip_angle)]) * norm


def magnetometer(rot: jax.Array, magvec: jax.Array) -> jax.Array:
    return maths.rotate(magvec, rot)


# Xsens MTI 10
# gyr:
# - bias error: 0.2 deg/s
# - bias stability: 18 deg/h
# - noise density: 0.03 deg/s/sqrt(hz)
# -> 40 hz: 0.2 deg/s
# -> 100 hz: 0.3 deg/s
# acc:
# - bias error: 0.05 m/s/s
# - bias stability: 15 micro g (<- gravity)
# - noise density: 60 micro g/sqrt(hz)
# -> 40 hz: 0.0036 m/s/s
# -> 100 hz: 0.006 m/s/s
# mag:
# - Total RMS noise: 0.5 milliGauss (1 Gauss = 1e-4 Tesla)
# -------------
# Xsens MTI 100
# gyr:
# - bias error: 0.2 deg/s
# - bias stability: 10 deg/h
# - noise density: 0.01 deg/s/sqrt(hz)
# -> 40 hz: 0.067 deg/s
# -> 100 hz: 0.1 deg/s
# acc:
# - bias error: 0.05 m/s/s
# - bias stability: 15 micro g (<- gravity)
# - noise density: 60 micro g/sqrt(hz)
# -> 40 hz: 0.0036 m/s/s
# -> 100 hz: 0.006 m/s/s
# mag:
# - Total RMS noise: 0.5 milliGauss
# -------------
# Movella Dot
# gyr:
# - bias error: ?
# - bias stability: 10 deg/h
# - noise density: 0.007 deg/s/sqrt(hz)
# acc:
# - bias error: ?
# - bias stability: 30 micro g
# - noise density: 120 micro g/sqrt(hz)
# mag:
# - Total RMS noise: 0.5 milliGauss = 5e-8 Tesla

# units are:
# - acc: m/s/s
# - gyr: rad/s
# - mag: a.u.
NOISE_LEVELS = {"acc": 0.048, "gyr": jnp.deg2rad(0.7), "mag": 0.01}
BIAS_LEVELS = {"acc": 0.5, "gyr": jnp.deg2rad(3), "mag": 0.0}


def add_noise_bias(
    key: jax.random.PRNGKey,
    imu_measurements: dict[str, jax.Array],
    noise_levels: Optional[dict[str, float | None]] = None,
    bias_levels: Optional[dict[str, float | None]] = None,
) -> dict[str, jax.Array]:
    """Add noise and bias to 6D or 9D imu measurements.

    Args:
        key (jax.random.PRNGKey): Random seed.
        imu_measurements (dict): IMU measurements without noise and bias.
            Format is {"gyr": Array, "acc": Array, "mag": Array}.

    Returns:
        dict: IMU measurements with noise and bias.
    """
    noise_levels = {} if noise_levels is None else noise_levels
    bias_levels = {} if bias_levels is None else bias_levels

    noisy_imu_measurements = {}
    for sensor in imu_measurements:
        key, c1, c2 = jax.random.split(key, 3)

        noise_scale = noise_levels.get(sensor, NOISE_LEVELS[sensor])
        if noise_scale is not None:
            noise = (
                jax.random.normal(c1, shape=imu_measurements[sensor].shape)
                * noise_scale
            )
        else:
            noise = 0.0

        bias_maxval = bias_levels.get(sensor, BIAS_LEVELS[sensor])
        if bias_maxval is not None:
            bias = jax.random.uniform(
                c2, minval=-bias_maxval, maxval=bias_maxval, shape=(3,)
            )
        else:
            bias = 0.0

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
    random_s2s_ori: Optional[float] = None,
    quasi_physical: bool = False,
    low_pass_filter_pos_f_cutoff: Optional[float] = None,
    low_pass_filter_rot_cutoff: Optional[float] = None,
    has_magnetometer: bool = False,
    magvec: Optional[jax.Array] = None,
    gyro_second_order: bool = False,
    natural_units: bool = False,
    acc_xinyuyi_n: int = 1,
) -> dict:
    """Simulates a 6D IMU, `xs` should be Transforms from eps-to-imu.
    NOTE: `smoothen_degree` is used as window size for moving average.
    NOTE: If `smoothen_degree` is given, and `delay` is not, then delay is chosen
    such moving average window is delayed to just be causal.
    """
    assert xs.ndim() == 2

    if random_s2s_ori is not None:
        assert key is not None, "`random_s2s_ori` requires a random seed via `key`"
        # `xs` are now from eps-to-segment, so add another final rotation from
        # segment-to-sensor where this transform is only rotational
        key, consume = jax.random.split(key)
        xs_s2s = base.Transform.create(
            rot=maths.quat_random(consume, maxval=random_s2s_ori)
        )
        xs = jax.vmap(algebra.transform_mul, in_axes=(None, 0))(xs_s2s, xs)

    if quasi_physical:
        xs = _quasi_physical_simulation(xs, dt)

    if low_pass_filter_pos_f_cutoff is not None:
        xs = xs.replace(
            pos=_butterworth(
                xs.pos, f_sampling=1 / dt, f_cutoff=low_pass_filter_pos_f_cutoff
            )
        )

    if low_pass_filter_rot_cutoff is not None:
        xs = xs.replace(
            rot=maths.quat_lowpassfilter(
                xs.rot, cutoff_freq=low_pass_filter_rot_cutoff, samp_freq=1 / dt
            )
        )

    measurements = {
        "acc": accelerometer(xs, gravity, dt, acc_xinyuyi_n),
        "gyr": gyroscope(xs.rot, dt, gyro_second_order),
    }

    if has_magnetometer:
        if magvec is None:
            assert key is not None
            key, consume = jax.random.split(key)
            magvec = _draw_random_magvec(consume)
        measurements["mag"] = magnetometer(xs.rot, magvec)

    if smoothen_degree is not None:
        measurements = jax.tree_map(
            lambda arr: _moving_average(arr, smoothen_degree),
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

    if natural_units:
        measurements = rescale_natural_units(measurements)

    return measurements


_rescale_natural_units_fns = defaultdict(lambda: (lambda arr: arr))
_rescale_natural_units_fns["gyr"] = lambda gyr: gyr / jnp.pi
_rescale_natural_units_fns["acc"] = lambda acc: acc / 9.81


def rescale_natural_units(X: dict[str, jax.Array]):
    return {key: _rescale_natural_units_fns[key](val) for key, val in X.items()}


def rel_pose(
    sys: base.System, xs: base.Transform, sys_xs: Optional[base.System] = None
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
        sys_xs = sys

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

        name_p = sys.idx_to_name(p)

        # find the transforms of those named bodies
        i = sys_xs.name_to_idx(name_i)
        p = sys_xs.name_to_idx(name_p)

        # get those transforms
        q1, q2 = xs.take(p).rot, xs.take(i).rot

        y[name_i] = qrel(q1, q2)

    sys.scan(pose_child_to_parent, "ll", sys.link_names, sys.link_parents)

    return y


def root_incl(
    sys: base.System,
    x: base.Transform,
    sys_x: base.System,
    child_to_parent: bool = False,
) -> dict[str, jax.Array]:
    # (time, nlinks, 4) -> (nlinks, time, 4)
    rots = x.rot.transpose((1, 0, 2))
    l_map = sys_x.idx_map("l")

    y = dict()

    def f(_, __, name: str, parent: int):
        if parent != -1:
            return
        q_i = maths.quat_project(rots[l_map[name]], jnp.array([0.0, 0, 1]))[1]
        if child_to_parent:
            q_i = maths.quat_inv(q_i)
        y[name] = q_i

    sys.scan(f, "ll", sys.link_names, sys.link_parents)

    return y


def root_full(
    sys: base.System,
    x: base.Transform,
    sys_x: base.System,
    child_to_parent: bool = False,
) -> dict[str, jax.Array]:
    # (time, nlinks, 4) -> (nlinks, time, 4)
    rots = x.rot.transpose((1, 0, 2))
    l_map = sys_x.idx_map("l")

    y = dict()

    def f(_, __, name: str, parent: int):
        if parent != -1:
            return
        q_i = rots[l_map[name]]
        if child_to_parent:
            q_i = maths.quat_inv(q_i)
        y[name] = q_i

    sys.scan(f, "ll", sys.link_names, sys.link_parents)

    return y


def joint_axes(
    sys: base.System,
    xs: base.Transform,
    sys_xs: base.System,
    key: Optional[jax.Array] = None,
    noisy: bool = False,
    from_sys: bool = False,
    randomly_flip: bool = False,
):
    """
    The joint-axes to world is always zeros.
    """
    if key is None:
        assert not noisy
        assert not randomly_flip

    N = xs.shape(axis=0)

    if from_sys:
        X = _joint_axes_from_sys(sys, N)
    else:
        X = _joint_axes_from_xs(sys, xs, sys_xs)

    if noisy:
        for name in X:
            key, c1, c2 = jax.random.split(key, 3)
            bias = maths.quat_random(c1, maxval=jnp.deg2rad(5.0))
            noise = maths.quat_random(c2, (N,), maxval=jnp.deg2rad(2.0))
            dist = maths.quat_mul(noise, bias)
            X[name]["joint_axes"] = maths.rotate(X[name]["joint_axes"], dist)

    # joint axes to world must be zeros
    for name, p in zip(sys.link_names, sys.link_parents):
        if p == -1:
            X[name]["joint_axes"] = jnp.zeros((N, 3))
        else:
            if randomly_flip:
                key, consume = jax.random.split(key)
                X[name]["joint_axes"] = (
                    jax.random.choice(consume, jnp.array([1.0, -1.0]))
                    * X[name]["joint_axes"]
                )

    return X


def _joint_axes_from_xs(sys, xs, sys_xs):

    xs = sim2real.match_xs(sys, xs, sys_xs)

    _, transform2_rot = sim2real.unzip_xs(sys, xs)
    qs = transform2_rot.rot.transpose((1, 0, 2))

    l2norm = lambda x: jnp.sqrt(jnp.sum(x**2, axis=-1))

    @jax.vmap
    def ensure_axis_convention(qs):
        axis = qs[..., 1:] / (
            jnp.linalg.norm(qs[..., 1:], axis=-1, keepdims=True) + 1e-6
        )
        convention = axis[0]
        cond = (l2norm(convention - axis) > l2norm(convention + axis))[..., None]
        return jnp.where(cond, -axis, axis)

    axes = ensure_axis_convention(qs)

    # TODO
    # not ideal to average vectors that live on a sphere
    N = axes.shape[1]
    axes_average = jnp.mean(axes, axis=1)
    axes_average /= jnp.linalg.norm(axes_average, axis=-1, keepdims=True)
    axes = jnp.repeat(axes_average[:, None], N, axis=1)

    X = {name: {"joint_axes": axes[sys.name_to_idx(name)]} for name in sys.link_names}
    return X


def _joint_axes_from_sys(sys: base.Transform, N: int) -> dict:
    "`sys` should be `sys_noimu`. `N` is number of timesteps"
    X = {}

    def f(_, __, name, link_type, link):
        joint_params = link.joint_params
        if link_type in ["rx", "ry", "rz"]:
            joint_axes = maths.unit_vectors(link_type[1])
        elif link_type == "rr":
            joint_axes = joint_params["rr"]["joint_axes"]
        elif link_type[:6] == "rr_imp":
            joint_axes = joint_params[link_type]["joint_axes"]
        else:
            joint_axes = maths.x_unit_vector
        X[name] = {"joint_axes": joint_axes}

    sys.scan(f, "lll", sys.link_names, sys.link_types, sys.links)
    X = jax.tree_map(lambda arr: jnp.repeat(arr[None], N, axis=0), X)
    return X


def _moving_average(arr: jax.Array, window: int) -> jax.Array:
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
    sys = io.load_sys_from_str(_quasi_physical_sys_str).replace(dt=dt)

    def step_dynamics(state: base.State, x):
        state = algorithms.step(sys.replace(link_spring_zeropoint=x.pos), state)
        return state, state.q

    state = base.State.create(sys, q=xs.pos[0])
    _, pos = jax.lax.scan(step_dynamics, state, xs)
    return xs.replace(pos=pos)


_constants = {
    "qp_damp": 35.0,
    "qp_stif": 625.0,
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


def _butterworth(
    signal: jax.Array,
    f_sampling: float,
    f_cutoff: int,
    method: str = "forward_backward",
) -> jax.Array:
    """https://stackoverflow.com/questions/20924868/calculate-coefficients-of-2nd-order
    -butterworth-low-pass-filter"""

    if method == "forward_backward":
        signal = _butterworth(signal, f_sampling, f_cutoff, "forward")
        return _butterworth(signal, f_sampling, f_cutoff, "backward")
    elif method == "forward":
        pass
    elif method == "backward":
        signal = jnp.flip(signal, axis=0)
    else:
        raise NotImplementedError

    ff = f_cutoff / f_sampling
    ita = 1.0 / jnp.tan(jnp.pi * ff)
    q = jnp.sqrt(2.0)
    b0 = 1.0 / (1.0 + q * ita + ita**2)
    b1 = 2 * b0
    b2 = b0
    a1 = 2.0 * (ita**2 - 1.0) * b0
    a2 = -(1.0 - q * ita + ita**2) * b0

    def f(carry, x_i):
        x_im1, x_im2, y_im1, y_im2 = carry
        y_i = b0 * x_i + b1 * x_im1 + b2 * x_im2 + a1 * y_im1 + a2 * y_im2
        return (x_i, x_im1, y_i, y_im1), y_i

    init = (signal[1], signal[0]) * 2
    signal = jax.lax.scan(f, init, signal[2:])[1]
    signal = jnp.concatenate((signal[0:1],) * 2 + (signal,))

    if method == "backward":
        signal = jnp.flip(signal, axis=0)

    return signal
