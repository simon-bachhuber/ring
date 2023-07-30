import jax

import x_xy


def test_jit_forward_kinematics():
    "This tests the lack of a certain bug. Details see function `_from_xml_vispy`"
    for sys in x_xy.io.list_load_examples():
        for _ in range(2):
            jax.jit(x_xy.algorithms.forward_kinematics)(
                sys, x_xy.base.State.create(sys)
            )
