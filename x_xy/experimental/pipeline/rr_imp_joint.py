import jax
import jax.numpy as jnp

import x_xy
from x_xy import maths
from x_xy.algorithms.jcalc import _draw_rxyz


def register_rr_imp_joint(
    config_res=x_xy.RCMG_Config(t_max=0.75), ang_max_deg: float = 5.0
):
    x_xy.update_n_joint_params(6)

    def _rr_imp_transform(q, params):
        axis_pri, axis_res = params[:3], params[3:]
        rot_pri = maths.quat_rot_axis(axis_pri, q[0])
        rot_res = maths.quat_rot_axis(axis_res, q[1])
        rot = x_xy.maths.quat_mul(rot_res, rot_pri)
        return x_xy.Transform.create(rot=rot)

    def _draw_rr_imp(config, key_t, key_value, _):
        key_t1, key_t2 = jax.random.split(key_t)
        key_value1, key_value2 = jax.random.split(key_value)
        q_traj_pri = _draw_rxyz(config, key_t1, key_value1, _)
        q_traj_res = _draw_rxyz(config_res, key_t2, key_value2, _)
        # scale to be within bounds
        q_traj_res = q_traj_res * (jnp.deg2rad(ang_max_deg) / jnp.pi)
        # center
        q_traj_res -= jnp.mean(q_traj_res)
        return jnp.concatenate((q_traj_pri[:, None], q_traj_res[:, None]), axis=1)

    rr_imp_joint = x_xy.JointModel(_rr_imp_transform, rcmg_draw_fn=_draw_rr_imp)
    try:
        x_xy.register_new_joint_type("rr_imp", rr_imp_joint, 2, 0)
    except AssertionError:
        pass


def setup_fn_randomize_joint_axes_primary_residual(
    key, sys: x_xy.System
) -> x_xy.System:
    joint_params = _draw_random_joint_axes(jax.random.split(key, sys.num_links()))
    return sys.replace(links=sys.links.replace(joint_params=joint_params))


@jax.vmap
def _draw_random_joint_axes(key):
    pri_axis = jnp.array([0, 0, 1.0])
    key1, key2 = jax.random.split(key)
    phi = jax.random.uniform(key1, maxval=2 * jnp.pi)
    res_axis = jnp.array([jnp.cos(phi), jnp.sin(phi), 0.0])
    random_rotation = maths.quat_random(key2)
    pri_axis = maths.rotate(pri_axis, random_rotation)
    res_axis = maths.rotate(res_axis, random_rotation)
    return jnp.concatenate((pri_axis, res_axis))
