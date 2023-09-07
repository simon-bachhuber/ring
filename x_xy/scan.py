from typing import Callable

import jax
import jax.numpy as jnp
import tree_utils

from . import base


def scan_sys(
    sys: base.System, f: Callable, in_types: str, *args, reverse: bool = False
):
    """Scan `f` along each link in system whilst carrying along state.

    Args:
        sys (base.System): _description_
        f (Callable[..., Y]): f(y: Y, *args) -> y
        in_types: string specifying the type of each input arg:
            'l' is an input to be split according to link ranges
            'q' is an input to be split according to q ranges
            'd' is an input to be split according to qd ranges
        args: Arguments passed to `f`, and split to match the link.
        reverse (bool, optional): If `true` from leaves to root. Defaults to False.

    Returns:
        ys: Stacked output y of f.
    """

    assert len(args) == len(in_types)

    order = range(sys.num_links())
    q_idx, qd_idx = 0, 0
    q_idxs, qd_idxs = {}, {}
    for link_idx, link_type in zip(order, sys.link_types):
        # build map from
        # link-idx -> q_idx
        # link-idx -> qd_idx
        q_idxs[link_idx] = slice(q_idx, q_idx + base.Q_WIDTHS[link_type])
        qd_idxs[link_idx] = slice(qd_idx, qd_idx + base.QD_WIDTHS[link_type])
        q_idx += base.Q_WIDTHS[link_type]
        qd_idx += base.QD_WIDTHS[link_type]

    idx_map = {
        "l": lambda link_idx: link_idx,
        "q": lambda link_idx: q_idxs[link_idx],
        "d": lambda link_idx: qd_idxs[link_idx],
    }

    if reverse:
        order = range(sys.num_links() - 1, -1, -1)

    y, ys = None, []
    for link_idx in order:
        args_link = [arg[idx_map[t](link_idx)] for arg, t in zip(args, in_types)]
        y = f(y, idx_map, *args_link)
        ys.append(y)

    if reverse:
        ys.reverse()

    ys = tree_utils.tree_batch(ys, backend="jax")
    return ys


def scan_links(sys: base.System, f: Callable, y0, *args, reverse: bool = False):
    """Scan `f` along each link in system whilst carrying along state.

    NOTE: TL;DR -> Don't use this for now. See below.

    NOTE: This function supports that `sys.link_parents` is a traced jax.Array.
    However, i could not get the `sys.link_joint_types` to follow the same logic.
    Or, the input dictionary `q` having a dynamic PyTreeDef.
    So, for now we settle with re-comilation, but this means we might as well
    use the simplified logic of `scan_links` function.

    Args:
        sys (base.System): _description_
        f (Callable[..., Y]): f(y: Y, *args) -> y
        y0: Initial carry state of type Y
        args: Arguments passed to `f`, and split to match the link.
        reverse (bool, optional): If `true` from leaves to root. Defaults to False.

    Returns:
        ys: Stacked output y of f.
    """

    def expand_mask(mask, shape):
        return mask[:, None] @ jnp.ones((1,) + shape, dtype=jnp.int32)

    def y_of_successors(mask_is_successor, ys):
        passed_links = len(ys)
        ys = tree_utils.tree_batch(ys, backend="jax")
        successor_ys = jax.tree_map(
            lambda a1, a2: jax.lax.select(
                expand_mask(mask_is_successor[-passed_links:], a2.shape[1:]), a1, a2
            ),
            ys,
            tree_utils.tree_batch(passed_links * [y0], backend="jax"),
        )

        get = lambda i: jax.tree_map(lambda arr: arr[i], successor_ys)
        y = get(0)
        for other_successors in range(1, tree_utils.tree_shape(successor_ys)):
            y = jax.tree_map(lambda a, b: a + b, y, get(other_successors))
        return y

    def y_of_predeccessor(predeccessor, ys):
        ys = tree_utils.tree_batch(ys, backend="jax")
        return jax.tree_map(lambda arr: arr[predeccessor], ys)

    successor_mask = lambda link: jnp.where(
        sys.parent == link, jnp.arange(sys.N), jnp.zeros(sys.N)
    ).astype(jnp.int32)
    predeccessor_idxs = lambda link: jax.lax.dynamic_index_in_dim(
        sys.parent, link, keepdims=False
    )

    order = range(sys.N)
    if reverse:
        order = range(sys.N - 1, -1, -1)

    y, ys = None, []
    for link_idx in order:
        if y is not None:
            if reverse:
                mask_is_successor = successor_mask(link_idx)
                y = jax.lax.cond(
                    jnp.sum(mask_is_successor) == 0,
                    lambda *args: y0,
                    y_of_successors,
                    mask_is_successor,
                    ys,
                )
            else:
                predeccessor = predeccessor_idxs(link_idx)
                y = jax.lax.cond(
                    predeccessor == -1,
                    lambda *args: y0,
                    y_of_predeccessor,
                    predeccessor,
                    ys,
                )
        else:
            y = y0

        args_link = [arg[link_idx] for arg in args]
        y = f(y, *args_link)

        if reverse:
            ys.insert(0, y)
        else:
            ys.append(y)

    ys = tree_utils.tree_batch(ys, backend="jax")
    return ys
