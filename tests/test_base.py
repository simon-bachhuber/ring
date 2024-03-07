from collections import defaultdict

import jax
import numpy as np
import ring


def load_ant():
    return ring.System(
        [-1, 0, 1, 0, 3, 0, 5, 0, 7],
        [],
        ["free"] + 8 * ["rx"],
        None,
        None,
        None,
        None,
        0.01,
        False,
        [[]],
    )


def test_tree():
    sys = load_ant()

    @jax.jit
    def forward(sys):
        qs, qds = [], []

        sum_of_link_idxs = defaultdict(lambda: 0.0)

        def f(_, __, parent, link, q, qd):
            sum_of_link_idxs[link] = sum_of_link_idxs[parent] + link
            qs.append(q)
            qds.append(qd)
            return sum_of_link_idxs[link]

        x = sys.scan(
            f,
            "llqd",
            sys.link_parents,
            np.arange(sys.num_links()),
            np.arange(sys.q_size()),
            np.arange(sys.qd_size()),
            reverse=False,
        )
        return x, qs, qds

    x, qs, qds = forward(sys)
    np.testing.assert_array_equal(x, np.array([0, 1, 3, 3, 7, 5, 11, 7, 15]))
    np.testing.assert_array_equal(qs[0], np.array(list(range(7))))
    np.testing.assert_array_equal(qs[1], np.array(list(range(7, 8))))
    np.testing.assert_array_equal(qds[0], np.array(list(range(6))))
    np.testing.assert_array_equal(qds[1], np.array(list(range(6, 7))))

    @jax.jit
    def reverse(sys):
        qs, qds = [], []

        sum_of_link_idxs = defaultdict(lambda: 0.0)

        def f(_, __, parent, link, q, qd):
            sum_of_link_idxs[link] = sum_of_link_idxs[link] + link
            sum_of_link_idxs[parent] = sum_of_link_idxs[parent] + sum_of_link_idxs[link]
            qs.append(q)
            qds.append(qd)
            return sum_of_link_idxs[link]

        x = sys.scan(
            f,
            "llqd",
            sys.link_parents,
            np.arange(sys.num_links()),
            np.arange(sys.q_size()),
            np.arange(sys.qd_size()),
            reverse=True,
        )
        return x, qs, qds

    x, qs, qds = reverse(sys)
    np.testing.assert_array_equal(x, np.array([36, 3, 2, 7, 4, 11, 6, 15, 8]))
    np.testing.assert_array_equal(qs[-1], np.array(list(range(7))))
    np.testing.assert_array_equal(qs[-2], np.array(list(range(7, 8))))
    np.testing.assert_array_equal(qds[-1], np.array(list(range(6))))
    np.testing.assert_array_equal(qds[-2], np.array(list(range(6, 7))))


def test_sys_idx_map():
    sys = ring.io.load_example("test_three_seg_seg2")

    idx_map_l = sys.idx_map("l")
    idx_map_q = sys.idx_map("q")
    idx_map_d = sys.idx_map("d")

    for name in sys.link_names:
        assert idx_map_l[name] == sys.name_to_idx(name)

    assert idx_map_q["seg2"] == slice(0, 7)
    assert idx_map_d["seg2"] == slice(0, 6)

    assert idx_map_q["seg1"] == slice(7, 8)
    assert idx_map_d["seg1"] == slice(6, 7)

    assert idx_map_q["imu1"] == slice(8, 8)
    assert idx_map_d["imu1"] == slice(7, 7)

    assert idx_map_q["seg3"] == slice(8, 9)
    assert idx_map_d["seg3"] == slice(7, 8)

    assert idx_map_q["imu2"] == slice(9, 9)
    assert idx_map_d["imu2"] == slice(8, 8)
