import jax
import numpy as np
import pytest

import x_xy


def sim(sys):
    state = x_xy.base.State.create(sys)
    for _ in range(100):
        state = jax.jit(x_xy.algorithms.step)(sys, state)
    return state.q


def test_sys_composer():
    sys1 = x_xy.io.load_example("three_segs/three_seg_seg2")
    sys2 = x_xy.io.load_example("test_double_pendulum")

    # these two systems are completely independent from another
    csys = x_xy.utils.inject_system(sys1, sys2)

    # thus forward simulation should be the same as before
    np.testing.assert_allclose(
        np.hstack((sim(sys1), sim(sys2))), sim(csys), atol=1e-5, rtol=100
    )

    assert csys.num_links() == sys1.num_links() + sys2.num_links()

    # names are duplicated
    with pytest.raises(AssertionError):
        csys = x_xy.utils.inject_system(sys2, sys2, "lower")

    # .. have to add a prefix
    csys = x_xy.utils.inject_system(sys2, sys2, "lower", prefix="sub_")
    assert len(sim(csys)) == csys.q_size() == 2 * sys2.q_size()
