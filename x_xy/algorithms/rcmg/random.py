import jax
import jax.numpy as jnp
from jax import random

from x_xy import maths


# APPROVED
def random_angle_over_time(
    key_t,
    key_ang,
    ANG_0,
    dang_min,
    dang_max,
    t_min,
    t_max,
    T,
    Ts,
    randomized_interpolation=False,
    range_of_motion=False,
    range_of_motion_method="uniform",
):
    def body_fn_outer(val):
        i, t, phi, key_t, key_ang, ANG = val

        key_t, consume = random.split(key_t)
        dt = random.uniform(consume, minval=t_min, maxval=t_max)
        t += dt

        key_ang, consume = random.split(key_ang)
        phi = _resolve_range_of_motion(
            range_of_motion,
            range_of_motion_method,
            dang_min,
            dang_max,
            dt,
            phi,
            consume,
        )

        ANG_i = jnp.array([[jnp.floor(t / Ts) * Ts, phi]])
        ANG = jax.lax.dynamic_update_slice_in_dim(ANG, ANG_i, start_index=i, axis=0)

        return i + 1, t, phi, key_t, key_ang, ANG

    def cond_fn_outer(val):
        i, t, phi, key_t, key_ang, ANG = val
        return t <= T

    # preallocate ANG array
    ANG = jnp.zeros((int(T // t_min) + 1, 2))
    ANG = ANG.at[0, 0].set(ANG_0)

    val_outer = (1, 0.0, ANG_0, key_t, key_ang, ANG)
    end, *_, consume, ANG = jax.lax.while_loop(cond_fn_outer, body_fn_outer, val_outer)
    ANG = jnp.where(
        (jnp.arange(len(ANG)) < end)[:, None],
        ANG,
        jax.lax.dynamic_index_in_dim(ANG, end - 1),
    )

    # resample
    t = jnp.arange(T, step=Ts)
    if randomized_interpolation:
        q = cosInterpolateRandomized()(t, ANG[:, 0], ANG[:, 1], consume)
    else:
        q = cosInterpolate(t, ANG[:, 0], ANG[:, 1])

    # if range_of_motion is true, then it is wrapped already
    if not range_of_motion:
        q = maths.wrap_to_pi(q)

    return q


# APPROVED
def random_position_over_time(
    key, POS_0, pos_min, pos_max, dpos_min, dpos_max, t_min, t_max, T, Ts, max_it
):
    def body_fn_inner(val):
        i, t, t_pre, x, x_pre, key = val

        def sample_dx(key):
            key, consume = random.split(key)
            dx = (
                random.uniform(consume) * (2 * dpos_max * t_max**2)
                - dpos_max * t_max**2
            )
            return key, dx

        key, dx = jax.lax.cond(i > max_it, (lambda key: (key, 0.0)), sample_dx, key)
        x = x_pre + dx

        return i + 1, t, t_pre, x, x_pre, key

    def cond_fn_inner(val):
        i, t, t_pre, x, x_pre, key = val
        dpos = abs((x - x_pre) / ((t - t_pre) ** 2))
        break_if_true1 = (
            (dpos < dpos_max) & (dpos > dpos_min) & (x >= pos_min) & (x <= pos_max)
        )
        break_if_true2 = i > max_it
        return ~(break_if_true1 | break_if_true2)

    def body_fn_outer(val):
        i, t, t_pre, x, x_pre, key, POS = val
        key, consume = random.split(key)
        t += random.uniform(consume, minval=t_min, maxval=t_max)

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

    # preallocate ANG array
    POS = jnp.zeros((int(T // t_min) + 1, 2))
    POS = POS.at[0, 0].set(POS_0)

    val_outer = (1, 0.0, 0.0, 0.0, 0.0, key, POS)
    end, *_, POS = jax.lax.while_loop(cond_fn_outer, body_fn_outer, val_outer)
    POS = jnp.where(
        (jnp.arange(len(POS)) < end)[:, None],
        POS,
        jax.lax.dynamic_index_in_dim(POS, end - 1),
    )

    # resample
    t = jnp.arange(T, step=Ts)
    r = cosInterpolate(t, POS[:, 0], POS[:, 1])
    return r


def _clip_to_pi(phi):
    return jnp.clip(phi, -jnp.pi, jnp.pi)


def _resolve_range_of_motion(
    range_of_motion, range_of_motion_method, dang_min, dang_max, dt, prev_phi, key
):
    key, consume = random.split(key)

    if range_of_motion:
        if range_of_motion_method == "coinflip":
            probs = jnp.array([0.5, 0.5])
        elif range_of_motion_method == "uniform":
            p = 0.5 * (1 - prev_phi / jnp.pi)
            probs = jnp.array([p, (1 - p)])
        else:
            raise NotImplementedError

        sign = random.choice(consume, jnp.array([1.0, -1.0]), p=probs)
        lower = _clip_to_pi(prev_phi + sign * dang_min * dt)
        upper = _clip_to_pi(prev_phi + sign * dang_max * dt)

        # swap if lower > upper
        lower, upper = jnp.sort(jnp.hstack((lower, upper)))

        key, consume = random.split(key)
        return random.uniform(consume, minval=lower, maxval=upper)

    else:
        dphi = random.uniform(consume, minval=dang_min, maxval=dang_max) * dt
        key, consume = random.split(key)
        sign = random.choice(consume, jnp.array([1.0, -1.0]))
        return prev_phi + sign * dphi


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


def _generate_cdf(cdf_bins):
    def __generate_cdf(key):
        samples = random.uniform(key, (cdf_bins,), maxval=1.0)
        samples = jnp.hstack((jnp.array([0.0]), samples))
        montonous = jnp.cumsum(samples)
        cdf = montonous / montonous[-1]
        return cdf

    return __generate_cdf


def cosInterpolateRandomized(cdf_bins=5):
    def _cosInterpolateRandomized(x, xp, fp, key):
        i = jnp.clip(jnp.searchsorted(xp, x, side="right"), 1, len(xp) - 1)
        dx = xp[i] - xp[i - 1]
        alpha = (x - xp[i - 1]) / dx

        key, *consume = random.split(key, len(xp) + 1)
        consume = jnp.array(consume).reshape((len(xp), 2))
        consume = consume[i - 1]
        cdfs = jax.vmap(_generate_cdf(cdf_bins))(consume)
        alpha = jax.vmap(_biject_alpha)(alpha, cdfs)

        def cos_interpolate(x1, x2, alpha):
            """x2 > x1"""
            return (x1 + x2) / 2 + (x1 - x2) / 2 * jnp.cos(alpha * jnp.pi)

        f = jnp.where(
            (dx == 0), fp[i], jax.vmap(cos_interpolate)(fp[i - 1], fp[i], alpha)
        )
        f = jnp.where(x > xp[-1], fp[-1], f)
        return f

    return _cosInterpolateRandomized
