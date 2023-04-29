from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from tree_utils import PyTree

from x_xy import base, maths, scan, utils
from x_xy.algorithms import (
    forward_kinematics_transforms,
    random_angle_over_time,
    random_position_over_time,
)


@dataclass
class RCMG_Config:
    T: float = 60.0  # length of random motion
    Ts: float = 0.01  # sampling rate
    t_min: float = 0.15  # min time between two generated angles
    t_max: float = 0.75  # max time ..

    dang_min: float = float(jnp.deg2rad(0))  # minimum angular velocity in deg/s
    dang_max: float = float(jnp.deg2rad(120))  # maximum angular velocity in deg/s

    # minimum angular velocity of euler angles used for `free and spherical joints`
    dang_min_free: float = float(jnp.deg2rad(0))
    dang_max_free: float = float(jnp.deg2rad(60))

    dpos_min: float = 0.001  # speed of translation
    dpos_max: float = 0.1
    pos_min: float = -2.5
    pos_max: float = +2.5

    # flags
    randomized_interpolation: bool = False
    range_of_motion_hinge: bool = True
    range_of_motion_hinge_method: str = "uniform"


def _draw_rxyz(
    config: RCMG_Config,
    key_t: jax.random.PRNGKey,
    key_value: jax.random.PRNGKey,
    enable_range_of_motion: bool = True,
) -> jax.Array:
    ANG_0 = 0.0
    return random_angle_over_time(
        key_t,
        key_value,
        ANG_0,
        config.dang_min,
        config.dang_max,
        config.t_min,
        config.t_max,
        config.T,
        config.Ts,
        config.randomized_interpolation,
        config.range_of_motion_hinge if enable_range_of_motion else False,
        config.range_of_motion_hinge_method,
    )


def _draw_pxyz(
    config: RCMG_Config, key_t: jax.random.PRNGKey, key_value: jax.random.PRNGKey
) -> jax.Array:
    POS_0 = 0.0
    max_iter = 100
    return random_position_over_time(
        key_value,
        POS_0,
        config.pos_min,
        config.pos_max,
        config.dpos_min,
        config.dpos_max,
        config.t_min,
        config.t_max,
        config.T,
        config.Ts,
        max_iter,
    )


def _draw_spherical(
    config: RCMG_Config, key_t: jax.random.PRNGKey, key_value: jax.random.PRNGKey
) -> jax.Array:
    # NOTE: We draw 3 euler angles and then build a quaternion.
    # Not ideal, but i am unaware of a better way.
    @jax.vmap
    def draw_euler_angles(key_t, key_value):
        return _draw_rxyz(config, key_t, key_value, enable_range_of_motion=False)

    triple = lambda key: jax.random.split(key, 3)
    euler_angles = draw_euler_angles(triple(key_t), triple(key_value)).T
    q = maths.quat_euler(euler_angles)
    return q


def _draw_free(
    config: RCMG_Config, key_t: jax.random.PRNGKey, key_value: jax.random.PRNGKey
) -> jax.Array:
    key_value1, key_value2 = jax.random.split(key_value)
    q = _draw_spherical(config, key_t, key_value1)
    pos = jax.vmap(lambda key: _draw_pxyz(config, None, key))(
        jax.random.split(key_value2, 3)
    ).T
    return jnp.concatenate((q, pos), axis=1)


def _draw_frozen(config: RCMG_Config, __, ___):
    N = int(config.T / config.Ts)
    return jnp.zeros((N, 0))


LINK_TYPE_TO_DRAW_Q_FN = {
    "rx": _draw_rxyz,
    "ry": _draw_rxyz,
    "rz": _draw_rxyz,
    "px": _draw_pxyz,
    "py": _draw_pxyz,
    "pz": _draw_pxyz,
    "free": _draw_free,
    "spherical": _draw_spherical,
    "frozen": _draw_frozen,
}


Generator = Callable[[jax.random.PRNGKey], PyTree]
SETUP_FN = Callable[[jax.random.PRNGKey, base.System], base.System]
FINALIZE_FN = Callable[[jax.Array, jax.Array, base.Transform, base.System], PyTree]


def build_generator(
    sys: base.System,
    config: RCMG_Config = RCMG_Config(),
    setup_fn: SETUP_FN = lambda key, sys: sys,
    finalize_fn: FINALIZE_FN = lambda key, q, x, sys: (q, x),
) -> Generator:
    def generator(key: jax.random.PRNGKey) -> dict:
        nonlocal sys
        # modified system
        key_start, consume = jax.random.split(key)
        sys_mod = setup_fn(consume, sys)

        # build generalized coordintes vector `q`
        q_list = []

        def draw_q(key, __, link_type):
            if key is None:
                key = key_start
            key, key_t, key_value = jax.random.split(key, 3)
            q_link = LINK_TYPE_TO_DRAW_Q_FN[link_type](config, key_t, key_value)
            # even revolute and prismatic joints must be 2d arrays
            q_link = q_link if q_link.ndim == 2 else q_link[:, None]
            q_list.append(q_link)
            return key

        key = scan.tree(sys_mod, draw_q, "l", sys.link_types)
        q = jnp.concatenate(q_list, axis=1)

        # do forward kinematics
        x, _ = jax.vmap(forward_kinematics_transforms, (None, 0))(sys_mod, q)

        return finalize_fn(key, q, x, sys_mod)

    return generator


def _build_batch_matrix(batchsizes: list[int]) -> jax.Array:
    arr = []
    for i, l in enumerate(batchsizes):
        arr += [i] * l
    return jnp.array(arr)


def batch_generator(
    generators: Generator | list[Generator], batchsizes: int | list[int]
) -> Generator:
    if not isinstance(generators, list):
        generators = [generators]
    if not isinstance(batchsizes, list):
        batchsizes = [batchsizes]

    assert len(generators) == len(batchsizes)

    batch_arr = _build_batch_matrix(batchsizes)
    bs_total = len(batch_arr)
    pmap, vmap = utils.distribute_batchsize(bs_total)
    batch_arr = batch_arr.reshape((pmap, vmap))

    @jax.pmap
    @jax.vmap
    def _generator(key, which_gen: int):
        return jax.lax.switch(which_gen, generators, key)

    def generator(key):
        pmap_vmap_keys = jax.random.split(key, bs_total).reshape((pmap, vmap, 2))
        data = _generator(pmap_vmap_keys, batch_arr)

        # merge pmap and vmap axis
        data = utils.merge_batchsize(data, pmap, vmap)

        return data

    return generator
