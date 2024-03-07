import jax.numpy as jnp
import ring
from ring import maths
from ring.algorithms.jcalc import _draw_rxyz
from ring.algorithms.jcalc import _p_control_term_rxyz
from ring.algorithms.jcalc import _qd_from_q_cartesian


def register_rr_joint():
    def _rr_transform(q, params):
        axis = params["joint_axes"]
        q = jnp.squeeze(q)
        rot = ring.maths.quat_rot_axis(axis, q)
        return ring.Transform.create(rot=rot)

    def _motion_fn(params):
        axis = params["joint_axes"]
        return ring.base.Motion.create(ang=axis)

    rr_joint = ring.JointModel(
        _rr_transform,
        motion=[_motion_fn],
        rcmg_draw_fn=_draw_rxyz,
        p_control_term=_p_control_term_rxyz,
        qd_from_q=_qd_from_q_cartesian,
        init_joint_params=_draw_random_joint_axis,
    )

    ring.register_new_joint_type("rr", rr_joint, 1, overwrite=True)


def _draw_random_joint_axis(key):
    return dict(joint_axes=maths.rotate(jnp.array([1.0, 0, 0]), maths.quat_random(key)))
