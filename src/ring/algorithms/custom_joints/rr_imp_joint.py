from dataclasses import replace

import jax
import jax.numpy as jnp

import ring
from ring import maths
from ring.algorithms.jcalc import _draw_rxyz
from ring.algorithms.jcalc import _p_control_term_rxyz
from ring.algorithms.jcalc import _qd_from_q_cartesian


def register_rr_imp_joint(
    config_res=ring.MotionConfig(dang_max=5.0, t_max=0.4),
    ang_max_deg: float = 7.5,
    name: str = "rr_imp",
):
    def _rr_imp_transform(q, params):
        axis_pri, axis_res = params["joint_axes"], params["residual"]
        rot_pri = maths.quat_rot_axis(axis_pri, q[0])
        rot_res = maths.quat_rot_axis(axis_res, q[1])
        rot = ring.maths.quat_mul(rot_res, rot_pri)
        return ring.Transform.create(rot=rot)

    def _draw_rr_imp(config, key_t, key_value, dt, N, _):
        key_t1, key_t2 = jax.random.split(key_t)
        key_value1, key_value2 = jax.random.split(key_value)
        q_traj_pri = _draw_rxyz(config, key_t1, key_value1, dt, N, _)
        q_traj_res = _draw_rxyz(
            replace(config_res, T=config.T), key_t2, key_value2, dt, N, _
        )
        # scale to be within bounds
        q_traj_res = q_traj_res * (jnp.deg2rad(ang_max_deg) / jnp.pi)
        # center
        q_traj_res -= jnp.mean(q_traj_res)
        return jnp.concatenate((q_traj_pri[:, None], q_traj_res[:, None]), axis=1)

    def _motion_fn_factory(whichone: str):
        def _motion_fn(params):
            axis = params[whichone]
            return ring.base.Motion.create(ang=axis)

        return _motion_fn

    rr_imp_joint = ring.JointModel(
        _rr_imp_transform,
        motion=[_motion_fn_factory("joint_axes"), _motion_fn_factory("residual")],
        rcmg_draw_fn=_draw_rr_imp,
        p_control_term=_p_control_term_rxyz,
        qd_from_q=_qd_from_q_cartesian,
        init_joint_params=_draw_random_joint_axes,
    )
    ring.register_new_joint_type(
        name,
        rr_imp_joint,
        2,
        2,
        overwrite=True,
    )


def _draw_random_joint_axes(key):
    pri_axis = jnp.array([0, 0, 1.0])
    key1, key2 = jax.random.split(key)
    phi = jax.random.uniform(key1, maxval=2 * jnp.pi)
    res_axis = jnp.array([jnp.cos(phi), jnp.sin(phi), 0.0])
    random_rotation = maths.quat_random(key2)
    pri_axis = maths.rotate(pri_axis, random_rotation)
    res_axis = maths.rotate(res_axis, random_rotation)
    return dict(joint_axes=pri_axis, residual=res_axis)
