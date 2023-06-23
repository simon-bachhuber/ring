from typing import Optional

import jax.numpy as jnp
from tree_utils import tree_batch

from x_xy import base, scan
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


def delete_subsystem(sys: base.System, link_name: str) -> base.System:
    "Cut subsystem starting at `link_name` (inclusive) from tree."
    subsys = _find_subsystem_indices(sys.link_parents, sys.name_to_idx(link_name))
    keep = jnp.array(list(set(range(sys.num_links())) - set(subsys)))

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
        take(sys.link_parents),
        sys.links[keep],
        take(sys.link_types),
        d,
        a,
        ss,
        sz,
        sys.dt,
        sys.dynamic_geometries,
        [geom for geom in sys.geoms if geom.link_idx in keep],
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
