import jax
import jax.numpy as jnp
import numpy as np
import pytest

import x_xy
from x_xy.utils import delete_subsystem, inject_system


def sim(sys):
    state = x_xy.base.State.create(sys)
    for _ in range(100):
        state = jax.jit(x_xy.algorithms.step)(sys, state)
    return state.q


def test_inject_system():
    sys1 = x_xy.io.load_example("three_segs/three_seg_seg2")
    sys2 = x_xy.io.load_example("test_double_pendulum")

    # these two systems are completely independent from another
    csys = inject_system(sys1, sys2)

    # thus forward simulation should be the same as before
    np.testing.assert_allclose(
        np.hstack((sim(sys1), sim(sys2))), sim(csys), atol=1e-5, rtol=100
    )

    assert csys.num_links() == sys1.num_links() + sys2.num_links()

    # names are duplicated
    with pytest.raises(AssertionError):
        csys = inject_system(sys2, sys2, "lower")

    # .. have to add a prefix
    csys = inject_system(sys2, sys2, "lower", prefix="sub_")
    assert len(sim(csys)) == csys.q_size() == 2 * sys2.q_size()


def test_delete_subsystem():
    sys1 = x_xy.io.load_example("three_segs/three_seg_seg2")
    sys2 = x_xy.io.load_example("test_double_pendulum")

    assert _tree_equal(delete_subsystem(inject_system(sys1, sys2), "upper"), sys1)
    assert _tree_equal(delete_subsystem(inject_system(sys2, sys1), "seg2"), sys2)
    assert _tree_equal(
        delete_subsystem(inject_system(sys2, sys1, at_body="upper"), "seg2"), sys2
    )


def test_tree_equal():
    sys = x_xy.io.load_example("three_segs/three_seg_seg2")
    sys_mod_nofield = sys.replace(link_parents=[i + 1 for i in sys.link_parents])
    sys_mod_field = sys.replace(link_damping=sys.link_damping + 1.0)

    with pytest.raises(AssertionError):
        assert _tree_equal(sys, sys_mod_nofield)

    with pytest.raises(AssertionError):
        assert _tree_equal(sys, sys_mod_field)

    assert _tree_equal(sys, sys)


def _tree_equal(a, b):
    "Copied from Marcel / Thomas"
    if type(a) is not type(b):
        return False
    if isinstance(a, x_xy.base._Base):
        return _tree_equal(a.__dict__, b.__dict__)
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(_tree_equal(a[k], b[k]) for k in a.keys())
    if isinstance(a, (tuple, list)):
        if len(a) != len(b):
            return False
        return all(_tree_equal(a[i], b[i]) for i in range(len(a)))
    if isinstance(a, jax.Array):
        return jnp.array_equal(a, b)
    return a == b
