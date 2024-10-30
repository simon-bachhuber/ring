import jax.numpy as jnp

import ring
from ring import maths
from ring.algorithms.jcalc import _draw_saddle
from ring.algorithms.jcalc import _p_control_term_rxyz
from ring.algorithms.jcalc import _qd_from_q_cartesian


def register_rsaddle_joint():
    def _transform(q, params):
        axes = params["joint_axes"]
        rot1 = maths.quat_rot_axis(axes[0], q[0])
        rot2 = maths.quat_rot_axis(axes[1], q[1])
        rot = maths.quat_mul(rot2, rot1)
        return ring.Transform.create(rot=rot)

    def _motion_fn_gen(i: int):
        def _motion_fn(params):
            axis = params["joint_axes"][i]
            return ring.base.Motion.create(ang=axis)

        return _motion_fn

    joint_model = ring.JointModel(
        _transform,
        motion=[_motion_fn_gen(i) for i in range(2)],
        rcmg_draw_fn=_draw_saddle,
        p_control_term=_p_control_term_rxyz,
        qd_from_q=_qd_from_q_cartesian,
        init_joint_params=_draw_random_joint_axes,
    )

    ring.register_new_joint_type("rsaddle", joint_model, 2, overwrite=True)


def _draw_random_joint_axes(key):
    return dict(
        joint_axes=maths.rotate(jnp.array([1.0, 0, 0]), maths.quat_random(key, (2,)))
    )
