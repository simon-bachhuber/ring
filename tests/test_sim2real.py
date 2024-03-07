from _compat import unbatch_gen
import jax
import jax.numpy as jnp
import numpy as np
import ring
from ring import maths
from ring import sim2real


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
        for eps_frame in [None, "inner"]:
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

            sys = ring.io.load_sys_from_str(sys_str)
            xs_omc = sim2real.xs_from_raw(sys, omc_data, qinv=qinv, eps_frame=eps_frame)
            t1_omc, t2_omc = sim2real.unzip_xs(sys, xs_omc)
            t1_sys = sys.links.transform1

            # t1_omc should be used when p == -1, else t1_sys
            @jax.vmap
            def merge_transform1(t1_omc):
                return jax.tree_map(
                    lambda a, b: jnp.where(
                        jnp.repeat(
                            jnp.array(sys.link_parents)[:, None] == -1,
                            a.shape[-1],
                            axis=-1,
                        ),
                        a,
                        b,
                    ),
                    t1_omc,
                    t1_sys,
                )

            t1 = merge_transform1(t1_omc)
            xs_sys = sim2real.zip_xs(sys, t1, t2_omc)

            np.testing.assert_allclose(xs_omc.take(0, 1).pos, xs_sys.take(0, 1).pos)
            np.testing.assert_allclose(xs_omc.take(0, 1).rot, xs_sys.take(0, 1).rot)

            outer_pos = []
            outer_rot = []

            for t in range(2):
                if eps_frame is not None:
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


def test_zip_unzip_scale():
    for sys in ring.io.list_load_examples():
        print(sys.model_name)
        _, xs = unbatch_gen(
            ring.algorithms.RCMG(
                sys, finalize_fn=lambda key, q, x, sys: (q, x)
            ).to_lazy_gen()
        )(
            jax.random.PRNGKey(
                1,
            )
        )

        t1, t2 = sim2real.unzip_xs(sys, xs)
        xs_re = sim2real.zip_xs(sys, t1, t2)

        jax.tree_map(
            lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-5),
            xs,
            xs_re,
        )

        xs_re = sim2real.scale_xs(sys, sim2real.scale_xs(sys, xs, 0.5), 2.0)
        np.testing.assert_allclose(xs.pos, xs_re.pos)
        np.testing.assert_allclose(
            maths.ensure_positive_w(xs.rot),
            maths.ensure_positive_w(xs_re.rot),
            rtol=1e-2,
            atol=1e-6,
        )
