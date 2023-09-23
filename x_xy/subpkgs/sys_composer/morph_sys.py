import jax
import jax.numpy as jnp

import x_xy
from x_xy import algebra
from x_xy import scan_sys
from x_xy.io import parse_system

from ... import base
from .identify_sys import identify_system
from .identify_sys import Node


def morph_system(sys: base.System, new_parents: list[int | str]) -> base.System:
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
                use = link_idx_old_indices

            pos_min_max_using_one = sys.links.transform1.pos.at[use].set(
                old_pos_min_max[use]
            )

            sys_mod = sys.replace(
                links=sys.links.replace(
                    transform1=sys.links.transform1.replace(pos=pos_min_max_using_one)
                )
            )
            pos_mod = _new_transform1(sys_mod, permutation, structure)[1][
                link_idx_old_indices
            ].pos

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
        new_parent_array,
        _permute(links).replace(
            joint_params=jnp.vstack([link[5] for link in _joint_properties])
        ),
        [link[4] for link in _joint_properties],
        stack_joint_properties(0),
        stack_joint_properties(1),
        stack_joint_properties(2),
        stack_joint_properties(3),
        sys.dt,
        sys.dynamic_geometries,
        _permute_modify_geoms(sys.geoms, structure),
        sys.gravity,
        sys.integration_method,
        sys.mass_mat_iters,
        _permute(sys.link_names),
        sys.model_name,
    )

    return parse_system(morphed_system)


def _new_transform1(
    sys: base.System,
    permutation: list[int],
    structure: list[Node],
    mod_geoms: bool = False,
    move_cs_one_up: bool = True,
):
    x = jax.jit(x_xy.forward_kinematics)(sys, x_xy.State.create(sys))[1].x

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
                    x_parent_to_this_node = x_xy.transform_mul(
                        x_this_node, x_xy.transform_inv(x_parent)
                    )
                    new_geoms = []
                    for geom in sys.geoms:
                        if geom.link_idx == node.link_idx_old_indices:
                            geom = geom.replace(
                                transform=x_xy.transform_mul(
                                    geom.transform, x_parent_to_this_node
                                )
                            )
                        new_geoms.append(geom)
                    sys = sys.replace(geoms=new_geoms)

    new_transform1s = sys.links.transform1
    for link_idx_old_indices in permutation:
        new_parent = structure[link_idx_old_indices].new_parent_old_indices
        if new_parent == -1:
            x_new_parent = x_xy.Transform.zero()
        else:
            x_new_parent = x_mod[new_parent]

        x_link = x_mod[link_idx_old_indices]
        new_transform1 = algebra.transform_mul(
            x_link, algebra.transform_inv(x_new_parent)
        )

        new_transform1s = new_transform1s.index_set(
            link_idx_old_indices, new_transform1
        )
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

    scan_sys(
        sys,
        filter_arrays,
        "dddq",
        sys.link_damping,
        sys.link_armature,
        sys.link_spring_stiffness,
        sys.link_spring_zeropoint,
    )
    return d, a, ss, sz


def _swapped_joint_properties(sys: base.System, structure: list[Node]):
    joint_properties = list(
        zip(*(_per_link_arrays(sys) + (sys.link_types, sys.links.joint_params)))
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
