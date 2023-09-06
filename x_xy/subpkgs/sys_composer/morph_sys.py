from typing import NamedTuple

import jax
import jax.numpy as jnp

from x_xy import algebra
from x_xy import base
from x_xy import parse_system
from x_xy import scan

from .identify_sys import identify_system
from .identify_sys import Neighbourhood


class JointProperties(NamedTuple):
    transform1: base.Transform
    pos_min: jax.Array
    pos_max: jax.Array
    link_type: str
    link_damping: jax.Array
    link_armature: jax.Array
    link_spring_stiffness: jax.Array
    link_spring_zeropoint: jax.Array

    def inv(self):
        return JointProperties(
            algebra.transform_inv(self.transform1),
            self.pos_max * -1.0,
            self.pos_min * -1.0,
            self.link_type,
            self.link_damping,
            self.link_armature,
            self.link_spring_stiffness,
            self.link_spring_zeropoint,
        )


def morph_system(
    sys: base.System, new_parents: list[int | str], prefix: str = ""
) -> base.System:
    """Re-orders the graph underlying the system. Returns a new system.

    Args:
        sys (base.System): System to be modified.
        new_parents (list[int]): Let the i-th entry have value j. Then, after morphing
            the system the system will be such that the link corresponding to the i-th
            link in the old system will have as parent the link corresponding to the
            j-th link in the old system.
        prefix (str): Prefix to prepend to all link names.

    Returns:
        base.System: Modified system.
    """

    structure, permutation, new_parent_array = identify_system(sys, new_parents)

    link_idx_old_world_old_indices = sys.link_parents.index(-1)
    d, a, ss, sz = _per_link_arrays(sys)

    def _get_joint_properties_of(i1: int, i2: int):
        return JointProperties(
            sys.links.transform1[i1],
            sys.links.pos_min[i1],
            sys.links.pos_max[i1],
            sys.link_types[i2],
            d[i2],
            a[i2],
            ss[i2],
            sz[i2],
        )

    joint_properties = []
    for neighbours in structure:
        if neighbours.new_parent_old_indices == -1:
            i1 = i2 = link_idx_old_world_old_indices
        elif neighbours.parent_changed:
            i1 = neighbours.new_parent_old_indices
        else:
            i1 = neighbours.link_idx_old_indices
        i2 = i1
        properties_link = _get_joint_properties_of(i1, i2)
        if neighbours.parent_changed:
            properties_link = properties_link.inv()
        joint_properties.append(properties_link)

    unpack = lambda attr: ([getattr(jp, attr) for jp in joint_properties])
    new_transform1 = unpack("transform1")
    new_transform1 = new_transform1[0].batch(*new_transform1[1:])
    new_pos_min = jnp.stack(unpack("pos_min"))
    new_pos_max = jnp.stack(unpack("pos_max"))

    new_links = sys.links.replace(
        transform1=new_transform1, pos_min=new_pos_min, pos_max=new_pos_max
    )

    new_links = _update_links_new_root_to_leaves(structure, permutation, new_links)

    new_link_types = unpack("link_type")
    d, a, ss, sz = map(
        unpack,
        (
            "link_damping",
            "link_armature",
            "link_spring_stiffness",
            "link_spring_zeropoint",
        ),
    )

    def _permute(obj):
        if isinstance(obj, (base._Base, jax.Array)):
            return obj[jnp.array(permutation, dtype=jnp.int32)]
        elif isinstance(obj, list):
            return [obj[permutation[i]] for i in range(len(obj))]
        assert False

    # permute those that have an indexing range not directly linked to 'l'
    d, a, ss, sz = map(lambda list: jnp.concatenate(_permute(list)), (d, a, ss, sz))

    morphed_system = base.System(
        new_parent_array,
        _permute(new_links),
        _permute(new_link_types),
        d,
        a,
        ss,
        sz,
        sys.dt,
        sys.dynamic_geometries,
        _permute_modify_geoms(sys.geoms, structure, new_transform1),
        sys.gravity,
        sys.integration_method,
        sys.mass_mat_iters,
        [prefix + name for name in _permute(sys.link_names)],
        sys.model_name,
    )

    return parse_system(morphed_system)


def _permute_modify_geoms(
    geoms: list[base.Geometry],
    structure: list[Neighbourhood],
    new_transform1: base.Transform,
) -> list[base.Geometry]:
    # change geom pointers & swap transforms
    geoms_mod = []
    for geom in geoms:
        if geom.link_idx != -1:
            neighbours = structure[geom.link_idx]
            if neighbours.parent_changed:
                transform = algebra.transform_mul(
                    geom.transform,
                    new_transform1[neighbours.link_idx_old_indices],
                )
                link_idx = neighbours.link_idx_new_indices
            else:
                transform = geom.transform
                link_idx = neighbours.link_idx_new_indices

            geom = geom.replace(
                link_idx=link_idx,
                transform=transform,
            )
        geoms_mod.append(geom)

    # then sort geoms in ascending order
    geoms_mod.sort(key=lambda geom: geom.link_idx)
    return geoms_mod


def _per_link_arrays(sys: base.System):
    d, a, ss, sz = [], [], [], []

    def filter_arrays(_, __, damp, arma, stiff, zero):
        d.append(damp)
        a.append(arma)
        ss.append(stiff)
        sz.append(zero)

    scan.tree(
        sys,
        filter_arrays,
        "dddq",
        sys.link_damping,
        sys.link_armature,
        sys.link_spring_stiffness,
        sys.link_spring_zeropoint,
    )
    return d, a, ss, sz


def _update_links_new_root_to_leaves(
    structure: list[Neighbourhood], permutation, new_links: base.Link
) -> base.Transform:
    del permutation

    transform1 = new_links.transform1
    new_transform1 = transform1
    pos_min = new_links.pos_min
    pos_max = new_links.pos_max

    for neigh in structure:
        if neigh.parent_changed:
            for child in structure:
                # if current link has direct childs with no parent changed
                # then update their transform1
                if child.new_parent_old_indices == neigh.link_idx_old_indices:
                    if not child.parent_changed:
                        transform1_child = algebra.transform_mul(
                            transform1[child.link_idx_old_indices],
                            transform1[neigh.link_idx_old_indices],
                        )
                        new_transform1 = new_transform1.index_set(
                            child.link_idx_old_indices, transform1_child
                        )

                        def shift_pos_min_max(pos):
                            return algebra.transform_mul(
                                base.Transform.create(pos=pos),
                                new_transform1[neigh.link_idx_old_indices],
                            ).pos

                        pos_min = pos_min.at[child.link_idx_old_indices].set(
                            shift_pos_min_max(pos_min[child.link_idx_old_indices])
                        )
                        pos_max = pos_max.at[child.link_idx_old_indices].set(
                            shift_pos_min_max(pos_max[child.link_idx_old_indices])
                        )

            # test if parent link connects to world
            parent = neigh.new_parent_old_indices
            if parent != -1:
                parent_parent = structure[parent].new_parent_old_indices
                if parent_parent == -1:
                    new_transform1 = new_transform1.index_set(
                        neigh.link_idx_old_indices, base.Transform.zero()
                    )

                    pos_min = pos_min.at[neigh.link_idx_old_indices].set(
                        jnp.zeros((3,))
                    )
                    pos_max = pos_max.at[neigh.link_idx_old_indices].set(
                        jnp.zeros((3,))
                    )

    return new_links.replace(
        transform1=new_transform1, pos_min=pos_min, pos_max=pos_max
    )
