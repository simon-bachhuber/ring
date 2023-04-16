import jax
import jax.numpy as jnp
import tree_utils
from flax import struct

from x_xy import base, scan


def empty_sys_TODO():
    return base.System([-1, 0, 1, 0, 3, 0, 5, 0, 7], [], [], 0.01, False)


def SKIP_test_scan_TODO():
    sys = empty_sys_TODO()

    qs = []

    def f(parent, link, q):
        if parent is not None:
            link += parent
        qs.append(q)
        return link

    ys = scan.scan_links(sys, f, list(range(sys.N)), list(range(sys.N)), reverse=True)

    link_indices = list(range(sys.N))
    link_indices.reverse()
    tree_utils.tree_close(
        (jnp.array(qs), ys),
        (jnp.array(link_indices), jnp.array([36, 3, 2, 7, 4, 11, 6, 15, 8])),
    )

    ys = scan.scan_links(sys, f, list(range(sys.N)), list(range(sys.N)), reverse=False)

    link_indices.reverse()
    tree_utils.tree_close(
        (qs, ys), (link_indices, jnp.array([0, 1, 3, 3, 7, 5, 11, 7, 15]))
    )

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


def test_scan_adv():
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
