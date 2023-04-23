from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from x_xy import algebra, base, maths

mrx = base.Motion.create(ang=jnp.array([1.0, 0, 0]))
mry = base.Motion.create(ang=jnp.array([0.0, 1, 0]))
mrz = base.Motion.create(ang=jnp.array([0.0, 0, 1]))
mpx = base.Motion.create(vel=jnp.array([1.0, 0, 0]))
mpy = base.Motion.create(vel=jnp.array([0.0, 1, 0]))
mpz = base.Motion.create(vel=jnp.array([0.0, 0, 1]))


@dataclass
class JointModel:
    # (q, params) -> Transform
    transform: Callable[[jax.Array, jax.Array], base.Transform]
    # len(motion) == len(qd)
    motion: list[base.Motion]


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


_joint_types = {
    "free": JointModel(_free_transform, [mrx, mry, mrz, mpx, mpy, mpz]),
    "frozen": JointModel(
        _frozen_transform,
        [],
    ),
    "spherical": JointModel(_spherical_transform, [mrx, mry, mrz]),
    "rx": JointModel(lambda q, _: _rxyz_transform(q, _, jnp.array([1.0, 0, 0])), [mrx]),
    "ry": JointModel(lambda q, _: _rxyz_transform(q, _, jnp.array([0.0, 1, 0])), [mry]),
    "rz": JointModel(lambda q, _: _rxyz_transform(q, _, jnp.array([0.0, 0, 1])), [mrz]),
    "px": JointModel(lambda q, _: _pxyz_transform(q, _, jnp.array([1.0, 0, 0])), [mpx]),
    "py": JointModel(lambda q, _: _pxyz_transform(q, _, jnp.array([0.0, 1, 0])), [mpy]),
    "pz": JointModel(lambda q, _: _pxyz_transform(q, _, jnp.array([0.0, 0, 1])), [mpz]),
}


def register_new_joint_type(
    joint_type: str,
    joint_model: JointModel,
    q_width: int,
    qd_width: Optional[int] = None,
):
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
