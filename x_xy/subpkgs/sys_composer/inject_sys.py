from typing import Optional

from tree_utils import tree_batch

from x_xy.io import parse_system

from ... import base


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
) -> base.System:
    """Combine two systems into one.

    Args:
        sys (base.System): Large system.
        sub_sys (base.System): Small system that will be included into the
            large system `sys`.
        at_body (Optional[str], optional): Into which body of the large system
            small system will be included. Defaults to `worldbody`.

    Returns:
        base.System: _description_
    """

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
        geoms=sys.geoms + sub_sys.geoms,
        gravity=sys.gravity,
        integration_method=sys.integration_method,
        mass_mat_iters=sys.mass_mat_iters,
        link_names=sys.link_names + sub_sys.link_names,
        model_name=sys.model_name,
    )

    return parse_system(combined_sys)
