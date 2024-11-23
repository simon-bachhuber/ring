from typing import Callable, Optional
import warnings

import jax
from jax import random
import jax.numpy as jnp

from ring import maths

Float = jax.Array
TimeDependentFloat = Callable[[Float], Float]


def _to_float(scalar: Float | TimeDependentFloat, t: Float) -> Float:
    if isinstance(scalar, Callable):
        return scalar(t)
    return scalar


# APPROVED
def random_angle_over_time(
    key_t: random.PRNGKey,
    key_ang: random.PRNGKey,
    ANG_0: float,
    dang_min: float | TimeDependentFloat,
    dang_max: float | TimeDependentFloat,
    delta_ang_min: float | TimeDependentFloat,
    delta_ang_max: float | TimeDependentFloat,
    t_min: float,
    t_max: float | TimeDependentFloat,
    T: float,
    Ts: float | jax.Array,
    N: Optional[int] = None,
    max_iter: int = 5,
    randomized_interpolation: bool = False,
    range_of_motion: bool = False,
    range_of_motion_method: str = "uniform",
    # this value has nothing to do with `range_of_motion` flag
    # this forces the value to stay within [ANG_0 - rom_halfsize, ANG_0 + rom_halfsize]
    rom_halfsize: float | TimeDependentFloat = 2 * jnp.pi,
    cdf_bins_min: int = 5,
    cdf_bins_max: Optional[int] = None,
    interpolation_method: str = "cosine",
) -> jax.Array:
    def body_fn_outer(val):
        i, t, phi, key_t, key_ang, ANG = val

        key_t, consume_t = random.split(key_t)
        key_ang, consume_ang = random.split(key_ang)
        rom_halfsize_float = _to_float(rom_halfsize, t)
        rom_lower = ANG_0 - rom_halfsize_float
        rom_upper = ANG_0 + rom_halfsize_float
        dt, phi = _resolve_range_of_motion(
            range_of_motion,
            range_of_motion_method,
            rom_lower,
            rom_upper,
            _to_float(dang_min, t),
            _to_float(dang_max, t),
            _to_float(delta_ang_min, t),
            _to_float(delta_ang_max, t),
            t_min,
            _to_float(t_max, t),
            phi,
            consume_t,
            consume_ang,
            max_iter,
        )
        t += dt

        # TODO do we really need the `jnp.floor(t / Ts) * Ts` since we resample later
        # anyways
        ANG_i = jnp.array([[jnp.floor(t / Ts) * Ts, phi]])
        ANG = jax.lax.dynamic_update_slice_in_dim(ANG, ANG_i, start_index=i, axis=0)

        return i + 1, t, phi, key_t, key_ang, ANG

    def cond_fn_outer(val):
        i, t, phi, key_t, key_ang, ANG = val
        return t <= T

    # preallocate ANG array
    _warn_huge_preallocation(t_min, T)
    ANG = jnp.zeros((int(T // t_min) + 1, 2))
    ANG = ANG.at[0, 1].set(ANG_0)

    val_outer = (1, 0.0, ANG_0, key_t, key_ang, ANG)
    end, *_, consume, ANG = jax.lax.while_loop(cond_fn_outer, body_fn_outer, val_outer)
    ANG = jnp.where(
        (jnp.arange(len(ANG)) < end)[:, None],
        ANG,
        jax.lax.dynamic_index_in_dim(ANG, end - 1),
    )

    # resample
    if N is None:
        t = jnp.arange(T, step=Ts)
    else:
        t = jnp.arange(N) * Ts
    if randomized_interpolation:
        q = interpolate(cdf_bins_min, cdf_bins_max, method=interpolation_method)(
            t, ANG[:, 0], ANG[:, 1], consume
        )
    else:
        if interpolation_method != "cosine":
            warnings.warn(
                f"You have select interpolation method {interpolation_method}. "
                "Differnt choices of interpolation method are only available if "
                "`randomized_interpolation` is set."
            )
        q = cosInterpolate(t, ANG[:, 0], ANG[:, 1])

    # if range_of_motion is true, then it is wrapped already
    if not range_of_motion:
        q = maths.wrap_to_pi(q)

    return q


# APPROVED
def random_position_over_time(
    key: random.PRNGKey,
    POS_0: float,
    pos_min: float | TimeDependentFloat,
    pos_max: float | TimeDependentFloat,
    dpos_min: float | TimeDependentFloat,
    dpos_max: float | TimeDependentFloat,
    t_min: float,
    t_max: float | TimeDependentFloat,
    T: float,
    Ts: float,
    N: Optional[int] = None,
    max_it: int = 100,
    randomized_interpolation: bool = False,
    cdf_bins_min: int = 5,
    cdf_bins_max: Optional[int] = None,
    interpolation_method: str = "cosine",
) -> jax.Array:
    def body_fn_inner(val):
        i, t, t_pre, x, x_pre, key = val
        dt = t - t_pre

        def sample_dx_squared(key):
            key, consume = random.split(key)
            dx = (
                random.uniform(consume) * (2 * dpos_max * t_max**2)
                - dpos_max * t_max**2
            )
            return key, dx

        def sample_dx(key):
            key, consume1, consume2 = random.split(key, 3)
            sign = random.choice(consume1, jnp.array([-1.0, 1.0]))
            dx = (
                sign
                * random.uniform(
                    consume2,
                    minval=_to_float(dpos_min, t_pre),
                    maxval=_to_float(dpos_max, t_pre),
                )
                * dt
            )
            return key, dx

        key, dx = jax.lax.cond(i > max_it, (lambda key: (key, 0.0)), sample_dx, key)
        x = x_pre + dx

        return i + 1, t, t_pre, x, x_pre, key

    def cond_fn_inner(val):
        i, t, t_pre, x, x_pre, key = val
        # this was used before as `dpos`, i don't know why i used a square here?
        # dpos = abs((x - x_pre) / ((t - t_pre) ** 2))  # noqa: F841
        dpos = jnp.abs((x - x_pre) / (t - t_pre))
        break_if_true1 = (
            (dpos < _to_float(dpos_max, t_pre))
            & (dpos > _to_float(dpos_min, t_pre))
            & (x >= _to_float(pos_min, t_pre))
            & (x <= _to_float(pos_max, t_pre))
        )
        break_if_true2 = i > max_it
        return jnp.logical_not(break_if_true1 | break_if_true2)

    def body_fn_outer(val):
        i, t, t_pre, x, x_pre, key, POS = val
        key, consume = random.split(key)
        t += random.uniform(consume, minval=t_min, maxval=_to_float(t_max, t_pre))

        # that zero resets the max_it count
        val_inner = (0, t, t_pre, x, x_pre, key)
        _, t, t_pre, x, x_pre, key = jax.lax.while_loop(
            cond_fn_inner, body_fn_inner, val_inner
        )

        POS_i = jnp.array([[jnp.floor(t / Ts) * Ts, x]])
        POS = jax.lax.dynamic_update_slice_in_dim(POS, POS_i, start_index=i, axis=0)
        t_pre = t
        x_pre = x
        return i + 1, t, t_pre, x, x_pre, key, POS

    def cond_fn_outer(val):
        i, t, t_pre, x, x_pre, key, POS = val
        return t <= T

    # preallocate POS array
    _warn_huge_preallocation(t_min, T)
    POS = jnp.zeros((int(T // t_min) + 1, 2))
    POS = POS.at[0, 1].set(POS_0)

    val_outer = (1, 0.0, 0.0, POS_0, POS_0, key, POS)
    end, *_, consume, POS = jax.lax.while_loop(cond_fn_outer, body_fn_outer, val_outer)
    POS = jnp.where(
        (jnp.arange(len(POS)) < end)[:, None],
        POS,
        jax.lax.dynamic_index_in_dim(POS, end - 1),
    )

    # resample
    if N is None:
        t = jnp.arange(T, step=Ts)
    else:
        t = jnp.arange(N) * Ts
    if randomized_interpolation:
        r = interpolate(cdf_bins_min, cdf_bins_max, method=interpolation_method)(
            t, POS[:, 0], POS[:, 1], consume
        )
    else:
        # TODO
        # Don't warn for position trajectories, i don't care about them as much
        if False:
            if interpolation_method != "cosine":
                warnings.warn(
                    f"You have select interpolation method {interpolation_method}. "
                    "Differnt choices of interpolation method are only available if "
                    "`randomized_interpolation` is set."
                )
        r = cosInterpolate(t, POS[:, 0], POS[:, 1])
    return r


_PREALLOCATION_WARN_LIMIT = 6000


def _warn_huge_preallocation(t_min, T):
    N = int(T // t_min) + 1
    if N > _PREALLOCATION_WARN_LIMIT:
        warnings.warn(
            f"The combination of `T`={T} and `t_min`={t_min} requires preallocating an "
            f"array with axis-length of {N} which is larger than the warn limit of "
            f"{_PREALLOCATION_WARN_LIMIT}. This might lead to large memory requirements"
            " and/or large jit-times, consider reducing `t_min`."
        )


def _clip_to_pi(phi):
    return jnp.clip(phi, -jnp.pi, jnp.pi)


def _resolve_range_of_motion(
    range_of_motion,
    range_of_motion_method,
    rom_lower: float,
    rom_upper: float,
    dang_min,
    dang_max,
    delta_ang_min,
    delta_ang_max,
    t_min,
    t_max,
    prev_phi,
    key_t,
    key_ang,
    max_iter,
):
    # legacy reasons, without range of motion the `sign` value, so going
    # left or right is 50-50 for free joints and spherical joints
    if not range_of_motion:
        range_of_motion_method = "coinflip"

    def _next_phi(key, dt):
        key, consume = random.split(key)

        if range_of_motion_method == "coinflip":
            probs = jnp.array([0.5, 0.5])
        elif range_of_motion_method == "uniform":
            p = 0.5 * (1 - prev_phi / jnp.pi)
            probs = jnp.array([p, (1 - p)])
        elif range_of_motion_method[:7] == "sigmoid":
            scale = 1.5
            provided_params = range_of_motion_method.split("-")
            if len(provided_params) == 2:
                scale = float(provided_params[-1])
            hardcut = jnp.pi - 0.01
            p = jnp.where(
                prev_phi > hardcut,
                0.0,
                jnp.where(prev_phi < -hardcut, 1.0, jax.nn.sigmoid(-scale * prev_phi)),
            )
            probs = jnp.array([p, (1 - p)])
        else:
            raise NotImplementedError

        sign = random.choice(consume, jnp.array([1.0, -1.0]), p=probs)
        lower = prev_phi + sign * dang_min * dt
        upper = prev_phi + sign * dang_max * dt

        if range_of_motion:
            lower, upper = _clip_to_pi(lower), _clip_to_pi(upper)

        # swap if lower > upper
        lower, upper = jnp.sort(jnp.hstack((lower, upper)))

        # clip bounds given by the angular velocity bounds to the rom bounds
        lower = jnp.clip(lower, a_min=rom_lower)
        upper = jnp.clip(upper, a_max=rom_upper)

        key, consume = random.split(key)
        return random.uniform(consume, minval=lower, maxval=upper)

    def body_fn(val):
        key_t, key_ang, _, _, i = val

        key_t, consume_t = jax.random.split(key_t)
        dt = jax.random.uniform(consume_t, minval=t_min, maxval=t_max)

        key_ang, consume_ang = jax.random.split(key_ang)
        next_phi = _next_phi(consume_ang, dt)

        return key_t, key_ang, dt, next_phi, i + 1

    def cond_fn(val):
        *_, dt, next_phi, i = val
        delta_phi = jnp.abs(next_phi - prev_phi)
        # delta_ang is in bounds
        cond_delta_ang = (delta_phi >= delta_ang_min) & (delta_phi <= delta_ang_max)
        # dang is in bounds
        dang = delta_phi / dt
        cond_dang = (dang >= dang_min) & (dang <= dang_max)

        break_if_true1 = jnp.logical_and(cond_delta_ang, cond_dang)
        # break out of loop
        break_if_true2 = i > max_iter
        return (i == 0) | (jnp.logical_not(break_if_true1 | break_if_true2))

    init_val = (key_t, key_ang, 1.0, prev_phi, 0)
    *_, dt, next_phi, _ = jax.lax.while_loop(cond_fn, body_fn, init_val)
    return dt, next_phi


def cosInterpolate(x, xp, fp):
    i = jnp.clip(jnp.searchsorted(xp, x, side="right"), 1, len(xp) - 1)
    dx = xp[i] - xp[i - 1]
    alpha = (x - xp[i - 1]) / dx

    def cos_interpolate(x1, x2, alpha):
        """x2 > x1"""
        return (x1 + x2) / 2 + (x1 - x2) / 2 * jnp.cos(alpha * jnp.pi)

    f = jnp.where((dx == 0), fp[i], jax.vmap(cos_interpolate)(fp[i - 1], fp[i], alpha))
    f = jnp.where(x > xp[-1], fp[-1], f)
    return f


def _biject_alpha(alpha, cdf):
    cdf_dx = 1 / (len(cdf) - 1)
    left_idx = (alpha // cdf_dx).astype(int)
    a = (alpha - left_idx * cdf_dx) / cdf_dx
    return (1 - a) * cdf[left_idx] + a * cdf[left_idx + 1]


def _generate_cdf(cdf_bins_min, cdf_bins_max=None):
    if cdf_bins_max is None:

        def _generate_cdf_min_eq_max(cdf_bins):
            def __generate_cdf(key):
                samples = random.uniform(key, (cdf_bins,), minval=1e-6, maxval=1.0)
                samples = jnp.hstack((jnp.array([0.0]), samples))
                montonous = jnp.cumsum(samples)
                cdf = montonous / montonous[-1]
                return cdf

            return __generate_cdf

        return _generate_cdf_min_eq_max(cdf_bins=cdf_bins_min)

    def _generate_cdf_min_uneq_max(dy_min, dy_max):
        assert dy_max >= dy_min

        def __generate_cdf(key):
            key, consume = random.split(key)
            cdf_bins = random.randint(consume, (), dy_min, dy_max + 1)
            mask = jnp.where(jnp.arange(dy_max) < cdf_bins, 1, 0)
            key, consume = random.split(key)
            mask = random.permutation(consume, mask)
            dy = random.uniform(key, (dy_max,), minval=1e-6, maxval=1.0)
            dy = dy[jnp.cumsum(mask) - 1]
            y = jnp.hstack((jnp.array([0.0]), dy))
            montonous = jnp.cumsum(y)
            cdf = montonous / montonous[-1]
            return cdf

        return __generate_cdf

    return _generate_cdf_min_uneq_max(cdf_bins_min, cdf_bins_max)


def interpolate(
    cdf_bins_min: int = 1, cdf_bins_max: Optional[int] = None, method: str = "cosine"
):
    "Interpolation with random alpha projection (disabled by default)."
    generate_cdf = _generate_cdf(cdf_bins_min, cdf_bins_max)

    def _interpolate(x, xp, fp, key):
        i = jnp.clip(jnp.searchsorted(xp, x, side="right"), 1, len(xp) - 1)
        dx = xp[i] - xp[i - 1]
        alpha = (x - xp[i - 1]) / dx

        key, *consume = random.split(key, len(xp) + 1)
        consume = jnp.array(consume).reshape((len(xp), 2))
        consume = consume[i - 1]
        cdfs = jax.vmap(generate_cdf)(consume)
        alpha = jax.vmap(_biject_alpha)(alpha, cdfs)

        def two_point_interp(x1, x2, alpha):
            """x2 > x1"""
            if method == "cosine":
                return (x1 + x2) / 2 + (x1 - x2) / 2 * jnp.cos(alpha * jnp.pi)
            elif method == "linear":
                return (1 - alpha) * x1 + alpha * x2
            else:
                raise NotImplementedError

        f = jnp.where(
            (dx == 0), fp[i], jax.vmap(two_point_interp)(fp[i - 1], fp[i], alpha)
        )
        f = jnp.where(x > xp[-1], fp[-1], f)
        return f

    return _interpolate
