import jax
import jax.numpy as jnp

import x_xy
from x_xy import maths
from x_xy.algorithms.jcalc import _draw_rxyz


def register_rr_joint():
    joint_params_pytree = dict(joint_axes=jnp.zeros((3,)))

    def _rr_transform(q, params):
        axis = params["joint_axes"]
        q = jnp.squeeze(q)
        rot = x_xy.maths.quat_rot_axis(axis, q)
        return x_xy.Transform.create(rot=rot)

    def _motion_fn(params):
        return x_xy.base.Motion.create(ang=params)

    rr_joint = x_xy.JointModel(
        _rr_transform, motion=[_motion_fn], rcmg_draw_fn=_draw_rxyz
    )

    x_xy.register_new_joint_type(
        "rr", rr_joint, 1, joint_params_pytree=joint_params_pytree, overwrite=True
    )


def setup_fn_randomize_joint_axes(key, sys: x_xy.System) -> x_xy.System:
    joint_axes = _draw_random_joint_axis(jax.random.split(key, sys.num_links()))
    sys.links.joint_params["rr"]["joint_axes"] = joint_axes
    return sys


@jax.vmap
def _draw_random_joint_axis(key):
    return maths.rotate(jnp.array([1.0, 0, 0]), maths.quat_random(key))
