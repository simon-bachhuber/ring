import _compat
import jax
import numpy as np
import pytest

import ring
from ring.sys_composer import delete_subsystem
from ring.sys_composer import identify_system
from ring.sys_composer import inject_system
from ring.sys_composer import morph_system
from ring.sys_composer.morph_sys import _autodetermine_new_parents
from ring.utils import sys_compare
from ring.utils import tree_equal


def sim(sys):
    state = ring.base.State.create(sys)
    for _ in range(100):
        state = jax.jit(ring.algorithms.step)(sys, state)
    return state.q


def test_inject_system():
    sys1 = ring.io.load_example("test_three_seg_seg2")
    sys2 = ring.io.load_example("test_double_pendulum")

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
    csys = inject_system(sys2, sys2.add_prefix_suffix("sub_"), "lower")
    assert len(sim(csys)) == csys.q_size() == 2 * sys2.q_size()


def test_delete_subsystem():
    sys1 = ring.io.load_example("test_three_seg_seg2")
    sys2 = ring.io.load_example("test_double_pendulum")

    assert tree_equal(delete_subsystem(inject_system(sys1, sys2), "upper"), sys1)
    assert tree_equal(delete_subsystem(inject_system(sys2, sys1), "seg2"), sys2)
    assert tree_equal(
        delete_subsystem(inject_system(sys2, sys1, at_body="upper"), "seg2"), sys2
    )

    # delete system "in the middle"
    sys3 = inject_system(
        inject_system(sys2, sys2.add_prefix_suffix("1")), sys2.add_prefix_suffix("2")
    )
    assert tree_equal(
        delete_subsystem(sys3, "1upper"),
        inject_system(sys2, sys2.add_prefix_suffix("2")),
    )

    # test jit
    jax.jit(delete_subsystem, static_argnums=1)(inject_system(sys1, sys2), "upper")


def test_delete_subsystem_cut_twice_versus_cut_once():
    # seg3 connects to -1
    sys = morph_system(
        ring.io.load_example("test_three_seg_seg2"),
        ["seg3", "seg2", "seg1", -1, "seg3"],
    )

    assert sys_compare(
        delete_subsystem(sys, ["seg1", "seg2"]), delete_subsystem(sys, ["seg2"])
    )

    sys = _compat._load_sys(1).morph_system(new_anchor="seg4")
    assert sys_compare(
        delete_subsystem(sys, ["seg2", "seg3"]), delete_subsystem(sys, ["seg3"])
    )


def test_tree_equal():
    sys = ring.io.load_example("test_three_seg_seg2")
    sys_mod_nofield = sys.replace(link_parents=[i + 1 for i in sys.link_parents])
    sys_mod_field = sys.replace(link_damping=sys.link_damping + 1.0)

    with pytest.raises(AssertionError):
        assert tree_equal(sys, sys_mod_nofield)

    with pytest.raises(AssertionError):
        assert tree_equal(sys, sys_mod_field)

    assert tree_equal(sys, sys)


def _load_sys(parent_arr: list[int]):
    return ring.base.System(
        parent_arr,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        link_names=[str(ele) for ele in parent_arr],
    )


def test_identify_system():
    list_equal = lambda l1, l2: all([e1 == e2 for e1, e2 in zip(l1, l2)])

    new_parents = [3, 0, 1, -1, 3]
    _, per, parent_array = identify_system(
        _load_sys(list(range(-1, 4))), new_parents, checks=False
    )
    assert list_equal(per, [3, 0, 1, 2, 4])
    assert list_equal(parent_array, [-1, 0, 1, 2, 0])

    # X---|----|
    #   0 O  5 O --- 6 O
    #    |-- 1 O --- 2 O
    #    |     |-- 3 O
    #    |-- 4 O
    parent_array = [-1, 0, 1, 1, 0, -1, 5]
    new_parents = [-1, 0, 1, 1, 0, -1, 5]
    _, per, parent_array = identify_system(
        _load_sys(parent_array), new_parents, checks=False
    )
    assert list_equal(per, list(range(7)))
    assert list_equal(parent_array, new_parents)

    # Node 3 connects to world
    # Node 5 connects to Node 0
    new_parents = [1, 3, 1, -1, 0, 0, 5]
    new_parent_array_truth = [-1, 0, 1, 2, 2, 4, 1]
    permutation_truth = [3, 1, 0, 4, 5, 6, 2]

    _, per, parent_array = identify_system(
        _load_sys(parent_array), new_parents, checks=False
    )
    assert list_equal(per, permutation_truth)
    assert list_equal(parent_array, new_parent_array_truth)


def SKIP_test_morph_all_examples():
    for example in ring.io.list_examples():
        print("Example: ", example)
        sys = ring.io.load_example(example)
        sys_re = morph_system(sys, sys.link_parents)
        # this should be a zero operation
        assert sys_compare(sys, sys_re)


def test_morph_one_example():
    for example in ring.io.list_examples()[:1]:
        print("Example: ", example)
        sys = ring.io.load_example(example)
        sys_re = morph_system(sys, sys.link_parents)
        # this should be a zero operation
        assert sys_compare(sys, sys_re)


def test_morph_four_seg():
    sys_seg1 = ring.io.load_example("test_morph_system/four_seg_seg1")
    sys_seg3 = ring.io.load_example("test_morph_system/four_seg_seg3")
    sys_seg3_from_seg1 = morph_system(
        sys_seg1, ["seg2", "seg3", -1, "seg3", "seg4", "seg1"]
    ).change_model_name(sys_seg3.model_name)
    assert sys_compare(sys_seg3, sys_seg3_from_seg1)


def test_autodetermine_new_parents():
    # arm.xml
    lam_arm = [-1, 0, 0, 2, 2, 4, 4, 6, 6, 8]
    solutions_arm = {
        0: lam_arm,
        1: [1, -1, 0, 2, 2, 4, 4, 6, 6, 8],
        2: [2, 0, -1, 2, 2, 4, 4, 6, 6, 8],
        4: [2, 0, 4, 2, -1, 4, 4, 6, 6, 8],
        6: [2, 0, 4, 2, 6, 4, -1, 6, 6, 8],
        8: [2, 0, 4, 2, 6, 4, 8, 6, -1, 8],
    }
    # gait.xml
    lam_gait = [-1, 0, 0, 2, 2, 4, 4, 6, 6, 8]
    solutions_gait = {
        0: lam_gait,
        2: [2, 0, -1, 2, 2, 4, 4, 6, 6, 8],
        4: [2, 0, 4, 2, -1, 4, 4, 6, 6, 8],
        6: [2, 0, 4, 2, 6, 4, -1, 6, 6, 8],
        8: [2, 0, 4, 2, 6, 4, 8, 6, -1, 8],
    }

    for lam, sol in zip([lam_arm, lam_gait], [solutions_arm, solutions_gait]):
        for anchor in sol:
            assert _autodetermine_new_parents(lam, anchor) == sol[anchor]


def test_morph_new_anchor():
    sys = _compat._load_sys(1)
    sys_compare(
        morph_system(
            sys,
            new_parents=[
                "seg2",
                "seg1",
                "seg3",
                "seg2",
                -1,
                "seg3",
                "seg3",
                "seg4",
                "seg4",
                "seg5",
            ],
        ),
        morph_system(sys, new_anchor="seg3"),
    )
