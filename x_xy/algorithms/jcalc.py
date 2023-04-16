from dataclasses import dataclass
from typing import Callable

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
    motion_subspace: Callable[[jax.Array], base.Motion]
    force_projection: Callable[[base.Force], jax.Array]


def _free_transform(q, _):
    rot, pos = q[:4], q[4:]
    # rot = maths.quat_euler(euler_angles)
    return base.Transform(pos, rot)


def _free_motion_subspace(qd):
    return base.Motion(jnp.ones((3,)) * qd[:3], jnp.ones((3,)) * qd[3:])


def _free_force_projection(f: base.Force) -> jax.Array:
    return jnp.array(
        [
            algebra.motion_dot(mrx, f),
            algebra.motion_dot(mry, f),
            algebra.motion_dot(mrz, f),
            algebra.motion_dot(mpx, f),
            algebra.motion_dot(mpy, f),
            algebra.motion_dot(mpz, f),
        ]
    )


def _rxyz_pxyz_force_projection(motion: base.Motion):
    def project(f: base.Force):
        return jnp.atleast_1d(algebra.motion_dot(motion, f))

    return project


def _rx_motion_subspace(qd):
    return mrx * qd


def _ry_motion_subspace(qd):
    return mry * qd


def _rz_motion_subspace(qd):
    return mrz * qd


def _px_motion_subspace(qd):
    return mpx * qd


def _py_motion_subspace(qd):
    return mpy * qd


def _pz_motion_subspace(qd):
    return mpz * qd


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
    "free": JointModel(_free_transform, _free_motion_subspace, _free_force_projection),
    "frozen": JointModel(
        _frozen_transform, lambda _: base.Motion.zero(), lambda _: jnp.array([0.0])
    ),
    "rx": JointModel(
        lambda q, _: _rxyz_transform(q, _, jnp.array([1.0, 0, 0])),
        _rx_motion_subspace,
        _rxyz_pxyz_force_projection(mrx),
    ),
    "ry": JointModel(
        lambda q, _: _rxyz_transform(q, _, jnp.array([0.0, 1, 0])),
        _ry_motion_subspace,
        _rxyz_pxyz_force_projection(mry),
    ),
    "rz": JointModel(
        lambda q, _: _rxyz_transform(q, _, jnp.array([0.0, 0, 1])),
        _rz_motion_subspace,
        _rxyz_pxyz_force_projection(mrz),
    ),
    "px": JointModel(
        lambda q, _: _pxyz_transform(q, _, jnp.array([1.0, 0, 0])),
        _px_motion_subspace,
        _rxyz_pxyz_force_projection(mpx),
    ),
    "py": JointModel(
        lambda q, _: _pxyz_transform(q, _, jnp.array([0.0, 1, 0])),
        _py_motion_subspace,
        _rxyz_pxyz_force_projection(mpy),
    ),
    "pz": JointModel(
        lambda q, _: _pxyz_transform(q, _, jnp.array([0.0, 0, 1])),
        _pz_motion_subspace,
        _rxyz_pxyz_force_projection(mpz),
    ),
}


def register_new_joint_type(joint_type: str, joint_model: JointModel):
    assert joint_type not in _joint_types, "already exists"
    _joint_types.update({joint_type: joint_model})


def jcalc_transform(joint_type: str, q: jax.Array, joint_params: jax.Array):
    return _joint_types[joint_type].transform(q, joint_params)


def jcalc_motion_subspace(joint_type: str, qd_qdd: jax.Array) -> base.Motion:
    return _joint_types[joint_type].motion_subspace(qd_qdd)


def jcalc_force_projection(joint_type: str, f: base.Force) -> jax.Array:
    return _joint_types[joint_type].force_projection(f)
