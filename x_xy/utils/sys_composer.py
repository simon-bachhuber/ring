import logging
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from tree_utils import tree_batch

from x_xy import algebra, base, scan
from x_xy.io import parse_system


# TODO
# right now this function won't really keep index ordering
# as one might expect.
# It simply appends the `sub_sys` at index end of `sys`, even
# though it might be injected in the middle of the index range
# of `sys`.
# This will be fixed once we have a `dump_sys_to_xml` function
def inject_system(
    sys: base.System,
    sub_sys: base.System,
    at_body: Optional[str] = None,
    prefix: str = "",
) -> base.System:
    """Combine two systems into one.

    Args:
        sys (base.System): Large system.
        sub_sys (base.System): Small system that will be included into the
            large system `sys`.
        at_body (Optional[str], optional): Into which body of the large system
            small system will be included. Defaults to `worldbody`.
        prefix (Optional[str], optional): Prefix that is added to body identifiers
            of small system. Defaults to ''.

    Returns:
        base.System: _description_
    """

    # give bodies new names if required
    sub_sys = sub_sys.replace(link_names=[prefix + name for name in sub_sys.link_names])

    # replace parent array
    if at_body is None:
        new_world = -1
    else:
        new_world = sys.name_to_idx(at_body)

    # append sub_sys at index end and replace sub_sys world with `at_body`
    N = sys.num_links()

    def new_parent(old_parent: int):
        if old_parent != -1:
            return old_parent + N
        else:
            return new_world

    sub_sys = sub_sys.replace(
        link_parents=[new_parent(p) for p in sub_sys.link_parents]
    )

    # replace link indices of geoms in sub_sys
    sub_sys = sub_sys.replace(
        geoms=[
            geom.replace(link_idx=new_parent(geom.link_idx)) for geom in sub_sys.geoms
        ]
    )

    # merge two systems
    concat = lambda a1, a2: tree_batch([a1, a2], True, "jax")
    combined_sys = base.System(
        link_parents=sys.link_parents + sub_sys.link_parents,
        links=concat(sys.links, sub_sys.links),
        link_types=sys.link_types + sub_sys.link_types,
        link_damping=concat(sys.link_damping, sub_sys.link_damping),
        link_armature=concat(sys.link_armature, sub_sys.link_armature),
        link_spring_stiffness=concat(
            sys.link_spring_stiffness, sub_sys.link_spring_stiffness
        ),
        link_spring_zeropoint=concat(
            sys.link_spring_zeropoint, sub_sys.link_spring_zeropoint
        ),
        dt=sys.dt,
        dynamic_geometries=sys.dynamic_geometries,
        geoms=sys.geoms + sub_sys.geoms,
        gravity=sys.gravity,
        integration_method=sys.integration_method,
        mass_mat_iters=sys.mass_mat_iters,
        link_names=sys.link_names + sub_sys.link_names,
        model_name=sys.model_name,
    )

    return parse_system(combined_sys)


def delete_subsystem(sys: base.System, link_name: str | list[str]) -> base.System:
    "Cut subsystem starting at `link_name` (inclusive) from tree."
    if isinstance(link_name, list):
        for ln in link_name:
            sys = delete_subsystem(sys, ln)
        return sys

    assert (
        link_name in sys.link_names
    ), f"link {link_name} not found in {sys.link_names}"

    subsys = _find_subsystem_indices(sys.link_parents, sys.name_to_idx(link_name))
    idx_map, keep = _idx_map_and_keepers(sys.link_parents, subsys)

    def take(list):
        return [ele for i, ele in enumerate(list) if i in keep]

    d, a, ss, sz = [], [], [], []

    def filter_arrays(_, __, damp, arma, stiff, zero, i: int):
        if i in keep:
            d.append(damp)
            a.append(arma)
            ss.append(stiff)
            sz.append(zero)

    scan.tree(
        sys,
        filter_arrays,
        "dddql",
        sys.link_damping,
        sys.link_armature,
        sys.link_spring_stiffness,
        sys.link_spring_zeropoint,
        list(range(sys.num_links())),
    )

    d, a, ss, sz = map(jnp.concatenate, (d, a, ss, sz))

    new_sys = base.System(
        _reindex_parent_array(sys.link_parents, subsys),
        sys.links[keep],
        take(sys.link_types),
        d,
        a,
        ss,
        sz,
        sys.dt,
        sys.dynamic_geometries,
        [
            geom.replace(link_idx=idx_map[geom.link_idx])
            for geom in sys.geoms
            if geom.link_idx in keep
        ],
        sys.gravity,
        sys.integration_method,
        sys.mass_mat_iters,
        take(sys.link_names),
        sys.model_name,
    )

    return parse_system(new_sys)


def _find_subsystem_indices(parents: list[int], k: int) -> list[int]:
    subsys = [k]
    for i, p in enumerate(parents):
        if p in subsys:
            subsys.append(i)
    return subsys


def _idx_map_and_keepers(parents: list[int], subsys: list[int]):
    num_links = len(parents)
    keep = jnp.array(list(set(range(num_links)) - set(subsys)))
    idx_map = dict(zip([-1] + keep.tolist(), range(-1, len(keep))))
    return idx_map, keep


def _reindex_parent_array(parents: list[int], subsys: list[int]) -> list[int]:
    idx_map, keep = _idx_map_and_keepers(parents, subsys)
    return [idx_map[p] for i, p in enumerate(parents) if i in keep]


class JointProperties(NamedTuple):
    transform1: base.Transform
    pos_min: jax.Array
    pos_max: jax.Array
    link_type: str
    link_damping: jax.Array
    link_armature: jax.Array
    link_spring_stiffness: jax.Array
    link_spring_zeropoint: jax.Array


def morph_system(
    sys: base.System, new_parents: list[int], prefix: str = ""
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

    # check that all `transform1` and `pos_min_max` of links that connect to -1
    # are set to zero / unity
    from x_xy.utils import tree_equal

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
    # does will not be preserved
    if sys.link_parents.count(-1) > 1:
        logging.warning(
            "Multiple bodies connect to worldbody. "
            "This ambiguity might not be preserved during morphing."
        )
    old_link_idx_to_world = sys.link_parents.index(-1)
    d, a, ss, sz = _per_link_arrays(sys)

    def _get_joint_properties_of(i: int):
        return JointProperties(
            sys.links.transform1[i],
            sys.links.pos_min[i],
            sys.links.pos_max[i],
            sys.link_types[i],
            d[i],
            a[i],
            ss[i],
            sz[i],
        )

    # permutation maps from new index to the old index, so e.g. at index position 0
    # is in the new system the link with index permutation[0] in the old system
    permutation, new_parent_array = _get_link_index_permutation_and_parent_array(
        new_parents
    )
    new_to_old_indices = permutation
    old_to_new_indices = _old_to_new(new_parents)

    def _permute(obj):
        if isinstance(obj, (base._Base, jax.Array)):
            return obj[jnp.array(permutation, dtype=jnp.int32)]
        elif isinstance(obj, list):
            return [obj[permutation[i]] for i in range(len(obj))]
        assert False

    # change geom pointers
    geoms = [
        geom.replace(
            link_idx=(old_to_new_indices[geom.link_idx] if geom.link_idx != -1 else -1)
        )
        for geom in sys.geoms
    ]
    # then sort geoms in ascending order
    geoms.sort(key=lambda geom: geom.link_idx)

    # swap between
    joint_properties = []
    for i in range(sys.num_links()):
        old_p = sys.link_parents[i]
        new_i = old_to_new_indices[i]
        new_p_new_indices = new_parent_array[new_i]
        new_p_old_indices = (new_to_old_indices + [-1])[new_p_new_indices]

        requires_inv = False
        if new_p_new_indices == -1:
            properties_of = old_link_idx_to_world
        elif old_p != new_p_old_indices:
            properties_of = new_p_old_indices
            requires_inv = True
        else:
            properties_of = i
        properties_i = _get_joint_properties_of(properties_of)

        if requires_inv:
            assert (
                sys.link_parents[new_p_old_indices] == i
            ), f"""I expexted parent-childs still to be connected with only
                their relative order inverted but link `{sys.idx_to_name(i)}`
                and `{sys.idx_to_name(new_p_old_indices)}` are not directly
                connected."""
            properties_i = _inv_properties(properties_i)
        joint_properties.append(properties_i)

    unpack = lambda attr: ([getattr(jp, attr) for jp in joint_properties])
    new_transform1 = unpack("transform1")
    new_transform1 = new_transform1[0].batch(*new_transform1[1:])
    new_pos_min = jnp.stack(unpack("pos_min"))
    new_pos_max = jnp.stack(unpack("pos_max"))
    new_links = sys.links.replace(
        transform1=new_transform1, pos_min=new_pos_min, pos_max=new_pos_max
    )
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

    # permute those that have an indexing range not directly linked to 'l'
    d, a, ss, sz = map(lambda list: jnp.concatenate(_permute(list)), (d, a, ss, sz))

    return base.System(
        new_parent_array,
        _permute(new_links),
        _permute(new_link_types),
        d,
        a,
        ss,
        sz,
        sys.dt,
        sys.dynamic_geometries,
        geoms,
        sys.gravity,
        sys.integration_method,
        sys.mass_mat_iters,
        _permute(sys.link_names),
        sys.model_name,
    )


def _get_link_index_permutation_and_parent_array(new_parents: list[int]):
    permutation = _new_to_old(new_parents)
    old_to_new_indices = _old_to_new(new_parents) + [-1]
    return permutation, [old_to_new_indices[new_parents[i]] for i in permutation]


def _new_to_old(new_parents: list[int]) -> list[int]:
    new_indices = []

    def find_childs_of(parent: int):
        for i, p in enumerate(new_parents):
            if p == parent:
                new_indices.append(i)
                find_childs_of(i)

    find_childs_of(-1)
    return new_indices


def _old_to_new(new_parents: list[int]) -> list[int]:
    old_to_new_indices = []
    new_to_old_indices = _new_to_old(new_parents)
    for new in range(len(new_parents)):
        old_to_new_indices.append(new_to_old_indices.index(new))
    return old_to_new_indices


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


def _inv_properties(prop: JointProperties):
    return JointProperties(
        algebra.transform_inv(prop.transform1),
        prop.pos_min * -1.0,
        prop.pos_max * -1.0,
        prop.link_type,
        prop.link_damping,
        prop.link_armature,
        prop.link_spring_stiffness,
        prop.link_spring_zeropoint,
    )
