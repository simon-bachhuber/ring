from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from ring import algebra
from ring import algorithms
from ring import base
from tree_utils import tree_batch


def _autodetermine_new_parents(lam: list[int], new_anchor: int) -> list[int]:
    "Automatically determines new parent array given a new anchor body."

    new_lam = {new_anchor: -1}

    def _connections(body: int, exclude: int | None) -> None:
        for i in range(len(lam)):
            if exclude is not None and i == exclude:
                continue

            if lam[i] == body or lam[body] == i:
                assert i not in new_lam
                new_lam[i] = body
                _connections(i, exclude=body)

    _connections(new_anchor, exclude=None)
    return [new_lam[i] for i in range(len(lam))]


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


class Node(NamedTuple):
    link_idx_old_indices: int
    link_idx_new_indices: int
    old_parent_old_indices: int
    old_parent_new_indices: int
    new_parent_old_indices: int
    new_parent_new_indices: int
    parent_changed: bool


def identify_system(
    sys: base.System, new_parents: list[int | str], checks: bool = True
) -> tuple[list[Node], list[int], list[int]]:
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
            Node(
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


def morph_system(
    sys: base.System,
    new_parents: Optional[list[int | str]] = None,
    new_anchor: Optional[int | str] = None,
) -> base.System:
    """Re-orders the graph underlying the system. Returns a new system.

    Args:
        sys (base.System): System to be modified.
        new_parents (list[int]): Let the i-th entry have value j. Then, after morphing
            the system the system will be such that the link corresponding to the i-th
            link in the old system will have as parent the link corresponding to the
            j-th link in the old system.

    Returns:
        base.System: Modified system.
    """

    assert not (new_parents is None and new_anchor is None)
    assert not (new_parents is not None and new_anchor is not None)

    if new_anchor is not None:
        if isinstance(new_anchor, str):
            new_anchor = sys.name_to_idx(new_anchor)
        new_parents = _autodetermine_new_parents(sys.link_parents, new_anchor)

    assert len(new_parents) == sys.num_links()

    structure, permutation, new_parent_array = identify_system(sys, new_parents)

    sys, new_transform1 = _new_transform1(sys, permutation, structure, True, True)

    def _new_pos_min_max(old_pos_min_max):
        new_pos_min_max = []
        for link_idx_old_indices in range(sys.num_links()):
            node = structure[link_idx_old_indices]
            if node.parent_changed and node.new_parent_old_indices != -1:
                grandparent = structure[
                    node.new_parent_old_indices
                ].new_parent_old_indices
                if grandparent != -1:
                    use = grandparent
                else:
                    # in this case we will always move the cs into the cs that connects
                    # to -1; thus the `pos_mod` will always be zeros no matter what we
                    # `use`
                    use = None
            else:
                use = link_idx_old_indices

            if use is not None:
                pos_min_max_using_one = sys.links.transform1.pos.at[use].set(
                    old_pos_min_max[use]
                )
            else:
                pos_min_max_using_one = sys.links.transform1.pos

            sys_mod = sys.replace(
                links=sys.links.replace(
                    transform1=sys.links.transform1.replace(pos=pos_min_max_using_one)
                )
            )

            # break early because we only use the value of `link_idx_old_indices` anways
            pos_mod = _new_transform1(
                sys_mod, permutation, structure, breakearly=link_idx_old_indices
            )[1][link_idx_old_indices].pos

            new_pos_min_max.append(pos_mod)
        return jnp.vstack(new_pos_min_max)

    new_pos_min_unsorted = _new_pos_min_max(sys.links.pos_min)
    new_pos_max_unsorted = _new_pos_min_max(sys.links.pos_max)
    new_pos_min = jnp.where(
        new_pos_min_unsorted > new_pos_max_unsorted,
        new_pos_max_unsorted,
        new_pos_min_unsorted,
    )
    new_pos_max = jnp.where(
        new_pos_max_unsorted < new_pos_min_unsorted,
        new_pos_min_unsorted,
        new_pos_max_unsorted,
    )
    links = sys.links.replace(
        transform1=new_transform1, pos_min=new_pos_min, pos_max=new_pos_max
    )

    def _permute(obj):
        if isinstance(obj, (base._Base, jax.Array)):
            return obj[jnp.array(permutation, dtype=jnp.int32)]
        elif isinstance(obj, list):
            return [obj[permutation[i]] for i in range(len(obj))]
        assert False

    _joint_properties = _permute(_swapped_joint_properties(sys, structure))
    stack_joint_properties = lambda i: jnp.concatenate(
        [link[i] for link in _joint_properties]
    )

    morphed_system = base.System(
        link_parents=new_parent_array,
        links=_permute(links).replace(
            joint_params=tree_batch(
                [link[5] for link in _joint_properties], backend="jax"
            )
        ),
        link_types=[link[4] for link in _joint_properties],
        link_damping=stack_joint_properties(0),
        link_armature=stack_joint_properties(1),
        link_spring_stiffness=stack_joint_properties(2),
        link_spring_zeropoint=stack_joint_properties(3),
        dt=sys.dt,
        geoms=_permute_modify_geoms(sys.geoms, structure),
        gravity=sys.gravity,
        integration_method=sys.integration_method,
        mass_mat_iters=sys.mass_mat_iters,
        link_names=_permute(sys.link_names),
        model_name=sys.model_name,
        omc=_permute(sys.omc),
    )

    return morphed_system.parse()


jit_for_kin = jax.jit(algorithms.forward_kinematics)


def _new_transform1(
    sys: base.System,
    permutation: list[int],
    structure: list[Node],
    mod_geoms: bool = False,
    move_cs_one_up: bool = True,
    breakearly: Optional[int] = None,
):
    x = jit_for_kin(sys, base.State.create(sys))[1].x

    # move all coordinate system of links with new parents "one up"
    # such that they are on top of the parents CS
    # but exclude if the new parent is -1
    x_mod = x
    if move_cs_one_up:
        for node in structure:
            if node.parent_changed and node.new_parent_old_indices != -1:
                x_this_node = x[node.link_idx_old_indices]
                x_parent = x[node.new_parent_old_indices]
                x_mod = x_mod.index_set(node.link_idx_old_indices, x_parent)

                if mod_geoms:
                    # compensate this transform for all geoms of this node
                    x_parent_to_this_node = algebra.transform_mul(
                        x_this_node, algebra.transform_inv(x_parent)
                    )
                    new_geoms = []
                    for geom in sys.geoms:
                        if geom.link_idx == node.link_idx_old_indices:
                            geom = geom.replace(
                                transform=algebra.transform_mul(
                                    geom.transform, x_parent_to_this_node
                                )
                            )
                        new_geoms.append(geom)
                    sys = sys.replace(geoms=new_geoms)

    new_transform1s = sys.links.transform1
    for link_idx_old_indices in permutation:
        new_parent = structure[link_idx_old_indices].new_parent_old_indices
        if new_parent == -1:
            x_new_parent = base.Transform.zero()
        else:
            x_new_parent = x_mod[new_parent]

        x_link = x_mod[link_idx_old_indices]
        new_transform1 = algebra.transform_mul(
            x_link, algebra.transform_inv(x_new_parent)
        )

        new_transform1s = new_transform1s.index_set(
            link_idx_old_indices, new_transform1
        )

        if breakearly == link_idx_old_indices:
            break

    return sys, new_transform1s


def _permute_modify_geoms(
    geoms: list[base.Geometry],
    structure: list[Node],
) -> list[base.Geometry]:
    # change geom pointers & swap transforms
    geoms_mod = []
    for geom in geoms:
        if geom.link_idx != -1:
            neighbours = structure[geom.link_idx]
            transform = geom.transform
            link_idx = neighbours.link_idx_new_indices

            geom = geom.replace(
                link_idx=link_idx,
                transform=transform,
            )
        geoms_mod.append(geom)
    return geoms_mod


def _per_link_arrays(sys: base.System):
    d, a, ss, sz = [], [], [], []

    def filter_arrays(_, __, damp, arma, stiff, zero):
        d.append(damp)
        a.append(arma)
        ss.append(stiff)
        sz.append(zero)

    sys.scan(
        filter_arrays,
        "dddq",
        sys.link_damping,
        sys.link_armature,
        sys.link_spring_stiffness,
        sys.link_spring_zeropoint,
    )
    return d, a, ss, sz


def _swapped_joint_properties(sys: base.System, structure: list[Node]) -> list:
    # convert joint_params from dict to list of dict; list if link-axis
    joint_params_list = [(sys.links[i]).joint_params for i in range(sys.num_links())]
    joint_properties = list(
        zip(*(_per_link_arrays(sys) + (sys.link_types, joint_params_list)))
    )

    swapped_joint_properties = []
    for node in structure:
        if node.new_parent_old_indices == -1:
            # find node that connects to world pre morph
            for swap_with_node in structure:
                if swap_with_node.old_parent_old_indices == -1:
                    break
            swap_with_node = swap_with_node.link_idx_old_indices
        else:
            if node.parent_changed:
                # use properties of parent then
                swap_with_node = node.new_parent_old_indices
            else:
                # otherwise nothing changed and no need to swap
                swap_with_node = node.link_idx_old_indices
        swapped_joint_properties.append(joint_properties[swap_with_node])
    return swapped_joint_properties
