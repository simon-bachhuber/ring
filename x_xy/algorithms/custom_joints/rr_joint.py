import jax.numpy as jnp

import x_xy
from x_xy import maths
from x_xy.algorithms.jcalc import _draw_rxyz
from x_xy.algorithms.jcalc import _p_control_term_rxyz
from x_xy.algorithms.jcalc import _qd_from_q_cartesian


def register_rr_joint():
    def _rr_transform(q, params):
        axis = params["joint_axes"]
        q = jnp.squeeze(q)
        rot = x_xy.maths.quat_rot_axis(axis, q)
        return x_xy.Transform.create(rot=rot)

    def _motion_fn(params):
        axis = params["joint_axes"]
        return x_xy.base.Motion.create(ang=axis)

    rr_joint = x_xy.JointModel(
        _rr_transform,
        motion=[_motion_fn],
        rcmg_draw_fn=_draw_rxyz,
        p_control_term=_p_control_term_rxyz,
        qd_from_q=_qd_from_q_cartesian,
        init_joint_params=_draw_random_joint_axis,
    )

    x_xy.register_new_joint_type("rr", rr_joint, 1, overwrite=True)


def _draw_random_joint_axis(key):
    return dict(joint_axes=maths.rotate(jnp.array([1.0, 0, 0]), maths.quat_random(key)))
