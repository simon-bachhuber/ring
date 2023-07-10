import jax
import numpy as np
import pytest

import x_xy
from x_xy.subpkgs.sys_composer import (
    _get_link_index_permutation_and_parent_array,
    delete_subsystem,
    inject_system,
    morph_system,
)
from x_xy.utils import tree_equal


def sim(sys):
    state = x_xy.base.State.create(sys)
    for _ in range(100):
        state = jax.jit(x_xy.algorithms.step)(sys, state)
    return state.q


def test_inject_system():
    sys1 = x_xy.io.load_example("test_three_seg_seg2")
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
    sys1 = x_xy.io.load_example("test_three_seg_seg2")
    sys2 = x_xy.io.load_example("test_double_pendulum")

    assert tree_equal(delete_subsystem(inject_system(sys1, sys2), "upper"), sys1)
    assert tree_equal(delete_subsystem(inject_system(sys2, sys1), "seg2"), sys2)
    assert tree_equal(
        delete_subsystem(inject_system(sys2, sys1, at_body="upper"), "seg2"), sys2
    )

    # delete system "in the middle"
    sys3 = inject_system(inject_system(sys2, sys2, prefix="1"), sys2, prefix="2")
    assert tree_equal(
        delete_subsystem(sys3, "1upper"), inject_system(sys2, sys2, prefix="2")
    )


def test_tree_equal():
    sys = x_xy.io.load_example("test_three_seg_seg2")
    sys_mod_nofield = sys.replace(link_parents=[i + 1 for i in sys.link_parents])
    sys_mod_field = sys.replace(link_damping=sys.link_damping + 1.0)

    with pytest.raises(AssertionError):
        assert tree_equal(sys, sys_mod_nofield)

    with pytest.raises(AssertionError):
        assert tree_equal(sys, sys_mod_field)

    assert tree_equal(sys, sys)


def test_morph_graph_functions():
    list_equal = lambda l1, l2: all([e1 == e2 for e1, e2 in zip(l1, l2)])

    new_parents = [3, 0, 1, -1, 3]
    per, parent_array = _get_link_index_permutation_and_parent_array(new_parents)
    assert list_equal(per, [3, 0, 1, 2, 4])
    assert list_equal(parent_array, [-1, 0, 1, 2, 0])

    # X---|----|
    #   0 O  5 O --- 6 O
    #    |-- 1 O --- 2 O
    #    |     |-- 3 O
    #    |-- 4 O
    new_parents = [-1, 0, 1, 1, 0, -1, 5]
    per, parent_array = _get_link_index_permutation_and_parent_array(new_parents)
    assert list_equal(per, list(range(7)))
    assert list_equal(parent_array, new_parents)

    # Node 3 connects to world
    # Node 5 connects to Node 0
    new_parents = [1, 3, 1, -1, 0, 0, 5]
    parent_array_truth = [-1, 0, 1, 2, 2, 4, 1]
    per_truth = [3, 1, 0, 4, 5, 6, 2]

    per, parent_array = _get_link_index_permutation_and_parent_array(new_parents)
    assert list_equal(per, per_truth)
    assert list_equal(parent_array, parent_array_truth)


def test_morph():
    exceptions = ["test_double_pendulum", "test_sensors", "branched"]
    for example in x_xy.io.list_examples():
        print("Example: ", example)
        sys = x_xy.io.load_example(example)

        if sys.model_name in exceptions:
            with pytest.raises(AssertionError):
                sys_re = morph_system(sys, sys.link_parents)
        else:
            sys_re = morph_system(sys, sys.link_parents)
            # this should be a zero operation
            assert tree_equal(sys, sys_re)

    # Test two known inverses
    sys = x_xy.io.load_example("test_three_seg_seg2")
    sys_re = morph_system(morph_system(sys, [1, 2, -1, 0, 3]), [1, 2, -1, 2, 3])
    assert tree_equal(sys, sys_re)

    sys = x_xy.io.load_example("test_kinematics")
    sys_re = morph_system(morph_system(sys, [1, -1, 0, 2, 2]), [1, -1, 1, 2, 2])
    assert tree_equal(sys, sys_re)


def test_morph_four_seg():
    sys_seg1 = x_xy.io.load_example("test_morph_system/four_seg_seg1")
    sys_seg2 = x_xy.io.load_example("test_morph_system/four_seg_seg2")
    sys_seg3 = x_xy.io.load_example("test_morph_system/four_seg_seg3")
    sys_seg3_from_seg2 = morph_system(sys_seg2, [3, 0, 1, -1, 3, 4]).change_model_name(
        sys_seg3.model_name
    )
    sys_seg1_from_seg2 = morph_system(sys_seg2, [1, -1, 1, 0, 3, 4]).change_model_name(
        sys_seg1.model_name
    )
    assert tree_equal(sys_seg3, sys_seg3_from_seg2)
    assert tree_equal(sys_seg1, sys_seg1_from_seg2)
