import jax
import jax.numpy as jnp

import x_xy
from x_xy import maths
from x_xy.algorithms.jcalc import _draw_rxyz


def register_rr_imp_joint(
    config_res=x_xy.RCMG_Config(dang_max=5.0, t_max=0.4), ang_max_deg: float = 7.5
):
    joint_params_pytree = dict(joint_axes=jnp.zeros((3,)), residual=jnp.zeros((3,)))

    def _rr_imp_transform(q, params):
        axis_pri, axis_res = params["joint_axes"], params["residual"]
        rot_pri = maths.quat_rot_axis(axis_pri, q[0])
        rot_res = maths.quat_rot_axis(axis_res, q[1])
        rot = x_xy.maths.quat_mul(rot_res, rot_pri)
        return x_xy.Transform.create(rot=rot)

    def _draw_rr_imp(config, key_t, key_value, dt, _):
        key_t1, key_t2 = jax.random.split(key_t)
        key_value1, key_value2 = jax.random.split(key_value)
        q_traj_pri = _draw_rxyz(config, key_t1, key_value1, dt, _)
        q_traj_res = _draw_rxyz(config_res, key_t2, key_value2, dt, _)
        # scale to be within bounds
        q_traj_res = q_traj_res * (jnp.deg2rad(ang_max_deg) / jnp.pi)
        # center
        q_traj_res -= jnp.mean(q_traj_res)
        return jnp.concatenate((q_traj_pri[:, None], q_traj_res[:, None]), axis=1)

    rr_imp_joint = x_xy.JointModel(_rr_imp_transform, rcmg_draw_fn=_draw_rr_imp)
    x_xy.register_new_joint_type(
        "rr_imp",
        rr_imp_joint,
        2,
        0,
        joint_params_pytree=joint_params_pytree,
        overwrite=True,
    )


def setup_fn_randomize_joint_axes_primary_residual(
    key, sys: x_xy.System
) -> x_xy.System:
    joint_params_rrimp = _draw_random_joint_axes(jax.random.split(key, sys.num_links()))
    sys.links.joint_params["rr_imp"] = joint_params_rrimp
    return sys


@jax.vmap
def _draw_random_joint_axes(key):
    pri_axis = jnp.array([0, 0, 1.0])
    key1, key2 = jax.random.split(key)
    phi = jax.random.uniform(key1, maxval=2 * jnp.pi)
    res_axis = jnp.array([jnp.cos(phi), jnp.sin(phi), 0.0])
    random_rotation = maths.quat_random(key2)
    pri_axis = maths.rotate(pri_axis, random_rotation)
    res_axis = maths.rotate(res_axis, random_rotation)
    return dict(joint_axes=pri_axis, residual=res_axis)
