from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import tree_utils
from flax import struct

from x_xy import base, scan


def load_ant():
    return base.System(
        [-1, 0, 1, 0, 3, 0, 5, 0, 7],
        [],
        ["free"] + 8 * ["rx"],
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

        x = scan.tree(
            sys,
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

        x = scan.tree(
            sys,
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


# old tests for `scan_links`


@struct.dataclass
class SystemAdv(base._Base):
    link_parents: jax.Array
    links: base.Link
    link_joint_types: list[str] = struct.field(False)

    # simulation timestep size
    dt: float = struct.field(False)

    # whether or not to re-calculate the inertia
    # matrix at every simulation timestep because
    # the geometries may have changed
    dynamic_geometries: bool = struct.field(False)

    # root / base acceleration offset
    gravity: jax.Array = jnp.array([0, 0, 9.81])

    @property
    def parent(self) -> jax.Array:
        return self.link_parents

    @property
    def N(self):
        return len(self.link_parents)


def empty_sys_adv():
    return SystemAdv(jnp.array([-1, 0, 1, 0, 3, 0, 5, 0, 7]), [], [], 0.01, False)


def SKIP_test_scan_adv():
    sys = empty_sys_adv()

    def f(state, link: int):
        _, parent = state
        link = jnp.array(link)
        return (link, link + parent)

    ys = scan.scan_links(
        sys, f, (jnp.array(0), jnp.array(0)), list(range(sys.N)), reverse=True
    )

    link_indices = jnp.arange(sys.N)
    tree_utils.tree_close(ys, (link_indices, jnp.array([36, 3, 2, 7, 4, 11, 6, 15, 8])))

    ys = scan.scan_links(
        sys, f, (jnp.array(0), jnp.array(0)), list(range(sys.N)), reverse=False
    )

    tree_utils.tree_close(ys, (link_indices, jnp.array([0, 1, 3, 3, 7, 5, 11, 7, 15])))

    qs = []

    def f(state, link: int, q):
        _, parent = state
        qs.append(q)
        link = jnp.array(link)
        return (link, link[None] + parent)

    @jax.jit
    def jit_scan(sys):
        ys = scan.scan_links(
            sys,
            f,
            (jnp.array(0), jnp.array([0])),
            list(range(sys.N)),
            list(range(sys.N)),
            reverse=False,
        )
        return ys

    ys = jit_scan(sys)

    tree_utils.tree_close(
        ys, (link_indices, jnp.array([[0], [1], [3], [3], [7], [5], [11], [7], [15]]))
    )

    assert tree_utils.tree_close(jnp.array(qs), link_indices)

    # weird that this works despite that impure .append inside `f`
    ys = jit_scan(sys)

    assert tree_utils.tree_close(jnp.array(qs), link_indices)
