from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import Callable, get_type_hints, Optional

import jax
import jax.numpy as jnp

from x_xy import algebra
from x_xy import base
from x_xy import maths

from ._random import random_angle_over_time
from ._random import random_position_over_time
from ._random import TimeDependentFloat


@dataclass
class RCMG_Config:
    T: float = 60.0  # length of random motion
    Ts: float = 0.01  # sampling rate
    t_min: float = 0.05  # min time between two generated angles
    t_max: float | TimeDependentFloat = 0.30  # max time ..

    dang_min: float | TimeDependentFloat = 0.1  # minimum angular velocity in rad/s
    dang_max: float | TimeDependentFloat = 3.0  # maximum angular velocity in rad/s

    # minimum angular velocity of euler angles used for `free and spherical joints`
    dang_min_free_spherical: float | TimeDependentFloat = 0.1
    dang_max_free_spherical: float | TimeDependentFloat = 3.0

    # max min allowed actual delta values in radians
    delta_ang_min: float | TimeDependentFloat = 0.0
    delta_ang_max: float | TimeDependentFloat = 2 * jnp.pi
    delta_ang_min_free_spherical: float | TimeDependentFloat = 0.0
    delta_ang_max_free_spherical: float | TimeDependentFloat = 2 * jnp.pi

    dpos_min: float | TimeDependentFloat = 0.001  # speed of translation
    dpos_max: float | TimeDependentFloat = 0.7
    pos_min: float | TimeDependentFloat = -2.5
    pos_max: float | TimeDependentFloat = +2.5

    # used by both `random_angle_*` and `random_pos_*`
    # only used if `randomized_interpolation` is set
    cdf_bins_min: int = 5
    # by default equal to `cdf_bins_min`
    cdf_bins_max: Optional[int] = None

    # flags
    randomized_interpolation_angle: bool = False
    randomized_interpolation_position: bool = False
    interpolation_method: str = "cosine"
    range_of_motion_hinge: bool = True
    range_of_motion_hinge_method: str = "uniform"

    # initial value of joints
    ang0_min: float = -jnp.pi
    ang0_max: float = jnp.pi
    pos0_min: float = 0.0
    pos0_max: float = 0.0

    # cor (center of rotation) custom fields
    cor_t_min: float = 0.2
    cor_t_max: float | TimeDependentFloat = 2.0
    cor_dpos_min: float | TimeDependentFloat = 0.00001
    cor_dpos_max: float | TimeDependentFloat = 0.5
    cor_pos_min: float | TimeDependentFloat = -0.4
    cor_pos_max: float | TimeDependentFloat = 0.4


def _find_interval(t: jax.Array, boundaries: jax.Array):
    """Find the interval of `boundaries` between which `t` lies.

    Args:
        t: Scalar float (e.g. time)
        boundaries: Array of floats

    Example: (from `test_jcalc.py`)
        >> _find_interval(1.5, jnp.array([0.0, 1.0, 2.0])) -> 2
        >> _find_interval(0.5, jnp.array([0.0])) -> 1
        >> _find_interval(-0.5, jnp.array([0.0])) -> 0
    """
    assert boundaries.ndim == 1

    @jax.vmap
    def leq_than_boundary(boundary: jax.Array):
        return jnp.where(t >= boundary, 1, 0)

    return jnp.sum(leq_than_boundary(boundaries))


def concat_configs(configs: list[RCMG_Config], boundaries: list[float]) -> RCMG_Config:
    assert len(configs) == (
        len(boundaries) + 1
    ), "length of `boundaries` should be one less than length of `configs`"
    boundaries = jnp.array(boundaries, dtype=float)

    def new_value(field: str):
        scalar_options = jnp.array([getattr(c, field) for c in configs])

        def scalar(t):
            return jax.lax.dynamic_index_in_dim(
                scalar_options, _find_interval(t, boundaries), keepdims=False
            )

        return scalar

    hints = get_type_hints(RCMG_Config())
    is_time_dependent_field = lambda key: hints[key] == (float | TimeDependentFloat)
    time_dependent_fields = [
        key for key in RCMG_Config().__dict__ if is_time_dependent_field(key)
    ]
    changes = {field: new_value(field) for field in time_dependent_fields}
    return replace(configs[0], **changes)


DRAW_FN = Callable[[RCMG_Config, jax.random.PRNGKey, jax.random.PRNGKey], jax.Array]


@dataclass
class JointModel:
    # (q, params) -> Transform
    transform: Callable[[jax.Array, jax.Array], base.Transform]
    # len(motion) == len(qd)
    motion: list[base.Motion] = field(default_factory=lambda: [])
    rcmg_draw_fn: Optional[DRAW_FN] = None


def _free_transform(q, _):
    rot, pos = q[:4], q[4:]
    return base.Transform(pos, rot)


def _rxyz_transform(q, _, axis):
    q = jnp.squeeze(q)
    rot = maths.quat_rot_axis(axis, q)
    return base.Transform.create(rot=rot)


def _pxyz_transform(q, _, direction):
    pos = direction * q
    return base.Transform.create(pos=pos)


def _frozen_transform(_, __):
    return base.Transform.zero()


def _spherical_transform(q, _):
    return base.Transform.create(rot=q)


def _p3d_transform(q, _):
    return base.Transform.create(pos=q)


mrx = base.Motion.create(ang=jnp.array([1.0, 0, 0]))
mry = base.Motion.create(ang=jnp.array([0.0, 1, 0]))
mrz = base.Motion.create(ang=jnp.array([0.0, 0, 1]))
mpx = base.Motion.create(vel=jnp.array([1.0, 0, 0]))
mpy = base.Motion.create(vel=jnp.array([0.0, 1, 0]))
mpz = base.Motion.create(vel=jnp.array([0.0, 0, 1]))


def _draw_rxyz(
    config: RCMG_Config,
    key_t: jax.random.PRNGKey,
    key_value: jax.random.PRNGKey,
    enable_range_of_motion: bool = True,
    free_spherical: bool = False,
) -> jax.Array:
    key_value, consume = jax.random.split(key_value)
    ANG_0 = jax.random.uniform(consume, minval=config.ang0_min, maxval=config.ang0_max)
    # `random_angle_over_time` always returns wrapped angles, thus it would be
    # inconsistent to allow an initial value that is not wrapped
    ANG_0 = maths.wrap_to_pi(ANG_0)
    # only used for `delta_ang_min_max` logic
    max_iter = 5
    return random_angle_over_time(
        key_t,
        key_value,
        ANG_0,
        config.dang_min_free_spherical if free_spherical else config.dang_min,
        config.dang_max_free_spherical if free_spherical else config.dang_max,
        config.delta_ang_min_free_spherical if free_spherical else config.delta_ang_min,
        config.delta_ang_max_free_spherical if free_spherical else config.delta_ang_max,
        config.t_min,
        config.t_max,
        config.T,
        config.Ts,
        max_iter,
        config.randomized_interpolation_angle,
        config.range_of_motion_hinge if enable_range_of_motion else False,
        config.range_of_motion_hinge_method,
        config.cdf_bins_min,
        config.cdf_bins_max,
        config.interpolation_method,
    )


def _draw_pxyz(
    config: RCMG_Config,
    _: jax.random.PRNGKey,
    key_value: jax.random.PRNGKey,
    cor: bool = False,
) -> jax.Array:
    key_value, consume = jax.random.split(key_value)
    POS_0 = jax.random.uniform(consume, minval=config.pos0_min, maxval=config.pos0_max)
    max_iter = 100
    return random_position_over_time(
        key_value,
        POS_0,
        config.cor_pos_min if cor else config.pos_min,
        config.cor_pos_max if cor else config.pos_max,
        config.cor_dpos_min if cor else config.dpos_min,
        config.cor_dpos_max if cor else config.dpos_max,
        config.cor_t_min if cor else config.t_min,
        config.cor_t_max if cor else config.t_max,
        config.T,
        config.Ts,
        max_iter,
        config.randomized_interpolation_position,
        config.cdf_bins_min,
        config.cdf_bins_max,
        config.interpolation_method,
    )


def _draw_spherical(
    config: RCMG_Config, key_t: jax.random.PRNGKey, key_value: jax.random.PRNGKey
) -> jax.Array:
    # NOTE: We draw 3 euler angles and then build a quaternion.
    # Not ideal, but i am unaware of a better way.
    @jax.vmap
    def draw_euler_angles(key_t, key_value):
        return _draw_rxyz(
            config, key_t, key_value, enable_range_of_motion=False, free_spherical=True
        )

    triple = lambda key: jax.random.split(key, 3)
    euler_angles = draw_euler_angles(triple(key_t), triple(key_value)).T
    q = maths.quat_euler(euler_angles)
    return q


def _draw_p3d_and_cor(
    config: RCMG_Config, _: jax.random.PRNGKey, key_value: jax.random.PRNGKey, cor: bool
) -> jax.Array:
    pos = jax.vmap(lambda key: _draw_pxyz(config, None, key, cor))(
        jax.random.split(key_value, 3)
    )
    return pos.T


def _draw_p3d(
    config: RCMG_Config, _: jax.random.PRNGKey, key_value: jax.random.PRNGKey
) -> jax.Array:
    return _draw_p3d_and_cor(config, _, key_value, cor=False)


def _draw_cor(
    config: RCMG_Config, _: jax.random.PRNGKey, key_value: jax.random.PRNGKey
) -> jax.Array:
    return _draw_p3d_and_cor(config, _, key_value, cor=True)


def _draw_free(
    config: RCMG_Config, key_t: jax.random.PRNGKey, key_value: jax.random.PRNGKey
) -> jax.Array:
    key_value1, key_value2 = jax.random.split(key_value)
    q = _draw_spherical(config, key_t, key_value1)
    pos = _draw_p3d(config, None, key_value2)
    return jnp.concatenate((q, pos), axis=1)


def _draw_frozen(config: RCMG_Config, __, ___):
    N = int(config.T / config.Ts)
    return jnp.zeros((N, 0))


_joint_types = {
    "free": JointModel(_free_transform, [mrx, mry, mrz, mpx, mpy, mpz], _draw_free),
    "frozen": JointModel(_frozen_transform, [], _draw_frozen),
    "spherical": JointModel(_spherical_transform, [mrx, mry, mrz], _draw_spherical),
    "p3d": JointModel(_p3d_transform, [mpx, mpy, mpz], _draw_p3d),
    "cor": JointModel(_p3d_transform, [mpx, mpy, mpz], _draw_cor),
    "rx": JointModel(
        lambda q, _: _rxyz_transform(q, _, jnp.array([1.0, 0, 0])), [mrx], _draw_rxyz
    ),
    "ry": JointModel(
        lambda q, _: _rxyz_transform(q, _, jnp.array([0.0, 1, 0])), [mry], _draw_rxyz
    ),
    "rz": JointModel(
        lambda q, _: _rxyz_transform(q, _, jnp.array([0.0, 0, 1])), [mrz], _draw_rxyz
    ),
    "px": JointModel(
        lambda q, _: _pxyz_transform(q, _, jnp.array([1.0, 0, 0])), [mpx], _draw_pxyz
    ),
    "py": JointModel(
        lambda q, _: _pxyz_transform(q, _, jnp.array([0.0, 1, 0])), [mpy], _draw_pxyz
    ),
    "pz": JointModel(
        lambda q, _: _pxyz_transform(q, _, jnp.array([0.0, 0, 1])), [mpz], _draw_pxyz
    ),
}


def register_new_joint_type(
    joint_type: str,
    joint_model: JointModel,
    q_width: int,
    qd_width: Optional[int] = None,
):
    if qd_width is None:
        qd_width = q_width

    assert len(joint_model.motion) == qd_width
    assert joint_type not in _joint_types, "already exists"
    _joint_types.update({joint_type: joint_model})
    base.Q_WIDTHS.update({joint_type: q_width})
    base.QD_WIDTHS.update({joint_type: qd_width})


def jcalc_transform(
    joint_type: str, q: jax.Array, joint_params: jax.Array
) -> base.Transform:
    return _joint_types[joint_type].transform(q, joint_params)


def jcalc_motion(joint_type: str, qd: jax.Array) -> base.Motion:
    list_motion = _joint_types[joint_type].motion
    m = base.Motion.zero()
    for dof in range(len(list_motion)):
        m += list_motion[dof] * qd[dof]
    return m


def jcalc_tau(joint_type: str, f: base.Force) -> jax.Array:
    list_motion = _joint_types[joint_type].motion
    return jnp.array([algebra.motion_dot(m, f) for m in list_motion])
