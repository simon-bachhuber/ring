import jax
import jax.numpy as jnp
import numpy as np

import x_xy
from x_xy import maths
from x_xy.io.omc.postprocess import forward_kinematics_omc, omc_to_xs


def test_forward_kinematics_omc():
    sys_str = r"""
    <x_xy>
    <options gravity="0 0 0" dt="0.01"/>
    <worldbody>
        <body name="inner" joint="free">
            <body name="outer" joint="rx" pos="1 0 1" euler="45 0 0"></body>
        </body>
    </worldbody>
    </x_xy>
    """

    for qinv in [False, True]:
        print(f"QInv: {qinv}")
        for eps_frame in ["none", None, "inner"]:
            print(f"Epsilon Frame: {eps_frame}")

            qrand = maths.quat_random(jax.random.PRNGKey(1), (2, 2))
            qrand_inv = maths.quat_inv(qrand)

            omc_data = {
                "inner": {
                    "pos": jnp.array([[1, 0, 0], [1, 2, 3]]),
                    "quat": qrand[:, 0],
                },
                "outer": {
                    "pos": jnp.ones((2, 3)),
                    "quat": qrand[:, 1],
                },
            }

            if qinv:
                qrand, qrand_inv = qrand_inv, qrand

            sys = x_xy.io.load_sys_from_str(sys_str)
            xs_omc = omc_to_xs(sys, omc_data, qinv=qinv, eps_frame=eps_frame)
            xs_sys = forward_kinematics_omc(sys, xs_omc)

            np.testing.assert_allclose(xs_omc.take(0, 1).pos, xs_sys.take(0, 1).pos)
            np.testing.assert_allclose(xs_omc.take(0, 1).rot, xs_sys.take(0, 1).rot)

            outer_pos = []
            outer_rot = []

            for t in range(2):
                if eps_frame != "none":
                    q_in_eps = qrand[t, 0]
                    q_in0_eps = qrand[0, 0]
                    q_in_in0 = maths.quat_mul(q_in_eps, maths.quat_inv(q_in0_eps))

                    pos_in_in0_in0 = maths.rotate(
                        omc_data["inner"]["pos"][t] - omc_data["inner"]["pos"][0],
                        q_in0_eps,
                    )

                    pos_out_in_in = sys.links[1].transform1.pos
                    q_joint_in = sys.links[1].transform1.rot
                    q_out_joint = maths.quat_mul(qrand[t, 1], qrand_inv[t, 0])

                    pos_out_in0_in0 = pos_in_in0_in0 + maths.rotate(
                        pos_out_in_in,
                        maths.quat_inv(q_in_in0),
                    )
                    rot_out_in0 = maths.quat_mul(
                        maths.quat_mul(q_out_joint, q_joint_in), q_in_in0
                    )
                    outer_pos.append(pos_out_in0_in0)
                    outer_rot.append(rot_out_in0)
                else:
                    pos_in_eps_eps = omc_data["inner"]["pos"][t]
                    q_in_eps = qrand[t, 0]

                    pos_out_in_in = sys.links[1].transform1.pos
                    q_joint_in = sys.links[1].transform1.rot
                    q_out_joint = maths.quat_mul(qrand[t, 1], qrand_inv[t, 0])

                    pos_out_eps_eps = pos_in_eps_eps + maths.rotate(
                        pos_out_in_in,
                        maths.quat_inv(q_in_eps),
                    )
                    rot_out_eps = maths.quat_mul(
                        maths.quat_mul(q_out_joint, q_joint_in), q_in_eps
                    )
                    outer_pos.append(pos_out_eps_eps)
                    outer_rot.append(rot_out_eps)

            np.testing.assert_allclose(
                jnp.vstack(outer_pos), xs_sys.take(1, 1).pos, atol=1e-6, rtol=1e-6
            )
            np.testing.assert_allclose(
                jnp.vstack(outer_rot), xs_sys.take(1, 1).rot, atol=1e-6, rtol=1e-6
            )
