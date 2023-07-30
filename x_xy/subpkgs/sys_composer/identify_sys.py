import logging
from typing import NamedTuple

import jax.numpy as jnp

from x_xy import base
from x_xy.utils import tree_equal


def _new_to_old_indices(new_parents: list[int]) -> list[int]:
    # aka permutation
    # permutation maps from new index to the old index, so e.g. at index position 0
    # is in the new system the link with index permutation[0] in the old system
    new_indices = []

    def find_childs_of(parent: int):
        for i, p in enumerate(new_parents):
            if p == parent:
                new_indices.append(i)
                find_childs_of(i)

    find_childs_of(-1)
    return new_indices + [-1]


def _old_to_new_indices(new_parents: list[int]) -> list[int]:
    old_to_new_indices = []
    new_to_old_indices = _new_to_old_indices(new_parents)
    for new in range(len(new_parents)):
        old_to_new_indices.append(new_to_old_indices.index(new))
    return old_to_new_indices + [-1]


class Neighbourhood(NamedTuple):
    link_idx_old_indices: int
    link_idx_new_indices: int
    old_parent_old_indices: int
    old_parent_new_indices: int
    new_parent_old_indices: int
    new_parent_new_indices: int
    parent_changed: bool


def identify_system(
    sys: base.System, new_parents: list[int | str], checks: bool = True
) -> list[Neighbourhood]:
    if checks:
        _identify_sys_checks(sys)

    new_parents_old_indices = [
        sys.name_to_idx(ele) if isinstance(ele, str) else ele for ele in new_parents
    ]
    new_to_old = _new_to_old_indices(new_parents_old_indices)
    old_to_new = _old_to_new_indices(new_parents_old_indices)

    structure = []
    for link_idx_old_indices in range(sys.num_links()):
        old_parent_old_indices = sys.link_parents[link_idx_old_indices]
        new_parent_old_indices = new_parents_old_indices[link_idx_old_indices]
        parent_changed = new_parent_old_indices != old_parent_old_indices
        structure.append(
            Neighbourhood(
                link_idx_old_indices,
                old_to_new[link_idx_old_indices],
                old_parent_old_indices,
                old_to_new[old_parent_old_indices],
                new_parent_old_indices,
                old_to_new[new_parent_old_indices],
                parent_changed,
            )
        )

        if checks and parent_changed and new_parent_old_indices != -1:
            assert (
                sys.link_parents[new_parent_old_indices] == link_idx_old_indices
            ), f"""I expexted parent-childs still to be connected with only their
                relative order inverted but link
                `{sys.idx_to_name(link_idx_old_indices)}` and
                `{sys.idx_to_name(new_parent_old_indices)}` are not directly
                connected."""

    # exclude the last value which is [-1]
    permutation = new_to_old[:-1]
    # order the list into a proper parents array
    new_parents_array_old_indices = [new_parents_old_indices[i] for i in permutation]

    return (
        structure,
        permutation,
        [old_to_new[p] for p in new_parents_array_old_indices],
    )


def _identify_sys_checks(sys: base.System) -> None:
    # check that all `transform1` and `pos_min_max` of links that connect to -1
    # are set to zero / unity
    for i in range(sys.num_links()):
        if sys.link_parents[i] == -1:
            link = sys.links[i]
            assertion_print = f"""Currently morphing systems with non-unity worldbody
            to link static transformations (e.g. through specifing `pos`, `euler`,
            `quat` in xml attributes of a body that directly connects to the
            worldbody) is not supported. The system `{sys.model_name}` and body
            `{sys.idx_to_name(i)}` violate this."""
            assert tree_equal(link.transform1, base.Transform.zero()), assertion_print
            assert tree_equal(link.pos_min, jnp.zeros((3,))), assertion_print
            assert tree_equal(link.pos_max, jnp.zeros((3,))), assertion_print

    # warn if there are joints to worldbody that are not of type `free`
    # those will not be preserved
    if sys.link_parents.count(-1) > 1:
        logging.warning(
            "Multiple bodies connect to worldbody. "
            "This ambiguity might not be preserved during morphing."
        )
