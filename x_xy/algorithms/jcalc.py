from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp

from x_xy import base, maths


@dataclass
class JointModel:
    # (q, params) -> Transform
    transform: Callable[[jax.Array, jax.Array], base.Transform]
    motion_subspace: Callable[[jax.Array, jax.Array], base.Motion]


def _free_transform(q, _):
    euler_angles, pos = q[:3], q[3:]
    rot = maths.quat_euler(euler_angles)
    return base.Transform(pos, rot)


def _free_motion_subspace(qd):
    return base.Motion(jnp.ones((3,)), jnp.ones((3,)))


_frozen_motion_subspace = base.Motion.zero()
_rx_motion_subspace = base.Motion.create(ang=jnp.array([1.0, 0, 0]))
_ry_motion_subspace = base.Motion.create(ang=jnp.array([0.0, 1, 0]))
_rz_motion_subspace = base.Motion.create(ang=jnp.array([0.0, 0, 1]))
_px_motion_subspace = base.Motion.create(vel=jnp.array([1.0, 0, 0]))
_py_motion_subspace = base.Motion.create(vel=jnp.array([0.0, 1, 0]))
_pz_motion_subspace = base.Motion.create(vel=jnp.array([0.0, 0, 1]))


def _frozen_transform(q, _):
    return base.Transform.zero()


def _rxyz_transform(q, _, axis):
    rot = maths.quat_rot_axis(axis, q)
    return base.Transform.create(rot=rot)


def _pxyz_transform(q, _, direction):
    pos = direction * q
    return base.Transform.create(pos=pos)


_joint_types = {
    # q is 6D
    # 3 Euler Angles in `xyz` convention
    # 3 Position Variables
    "free": JointModel(_free_transform, _free_motion_subspace),
    "frozen": JointModel(_frozen_transform, _frozen_motion_subspace),
    "rx": JointModel(
        lambda q, _: _rxyz_transform(q, _, jnp.array([1.0, 0, 0])), _rx_motion_subspace
    ),
    "ry": JointModel(
        lambda q, _: _rxyz_transform(q, _, jnp.array([0.0, 1, 0])), _ry_motion_subspace
    ),
    "rz": JointModel(
        lambda q, _: _rxyz_transform(q, _, jnp.array([0.0, 0, 1])), _rz_motion_subspace
    ),
    "px": JointModel(
        lambda q, _: _pxyz_transform(q, _, jnp.array([1.0, 0, 0])), _px_motion_subspace
    ),
    "py": JointModel(
        lambda q, _: _pxyz_transform(q, _, jnp.array([0.0, 1, 0])), _py_motion_subspace
    ),
    "pz": JointModel(
        lambda q, _: _pxyz_transform(q, _, jnp.array([0.0, 0, 1])), _pz_motion_subspace
    ),
}


def register_new_joint_type(joint_type: str, joint_model: JointModel):
    assert joint_type not in _joint_types, "already exists"
    _joint_types.update({joint_type: joint_model})


def jcalc_transform(joint_type: str, q: jax.Array, joint_params: jax.Array):
    return _joint_types[joint_type].transform(q, joint_params)


def jcalc_motion_subspace(joint_type: str) -> base.Motion:
    return _joint_types[joint_type].motion_subspace
