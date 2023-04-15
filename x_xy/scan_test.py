import jax
import jax.numpy as jnp
import tree_utils

from x_xy import base, scan


def empty_sys():
    return base.System(jnp.array([-1, 0, 1, 0, 3, 0, 5, 0, 7]), [], [], 0.01, False)


def test_scan():
    sys = empty_sys()

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
