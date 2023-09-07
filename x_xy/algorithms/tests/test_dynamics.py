import jax
import jax.numpy as jnp
import numpy as np

import x_xy

xml_str_2_link_w_inertia = r"""
<x_xy>
    <options gravity="0 0 9.81" dt="0.01"/>
    <defaults>
        <geom mass="10" pos="1 1 1" dim="1 1 1" type="box"/>
    </defaults>
    <worldbody>
        <body name="upper" pos="0 1 0" joint="rx">
            <geom/>
            <body name="lower" pos="0 1 0" joint="rz">
                <geom/>
            </body>
        </body>
    </worldbody>
</x_xy>
"""


def test_inverse_dynamics_and_mass_matrix():
    sys = x_xy.io.load_sys_from_str(xml_str_2_link_w_inertia)
    q = jnp.array([-jnp.pi / 2, 0])
    qd = jnp.array([0.0, 0])
    qdd = qd
    state = x_xy.State.create(sys, q, qd)
    sys, state = x_xy.forward_kinematics(sys, state)

    C = jax.jit(x_xy.inverse_dynamics)(sys, state.qd, qdd)
    H = jax.jit(x_xy.compute_mass_matrix)(sys)

    np.testing.assert_allclose(
        C, np.array([196.20001, -98.100006], dtype=np.float32), atol=1e-4, rtol=1e-6
    )
    np.testing.assert_allclose(
        H,
        np.array([[73.33333, -10.0], [-10.0, 21.66666]], dtype=np.float32),
        atol=1e-5,
        rtol=1e-7,
    )
