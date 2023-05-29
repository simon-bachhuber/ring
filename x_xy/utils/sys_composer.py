from typing import Optional

from tree_utils import tree_batch

from x_xy import base
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
    sub_sys = sub_sys.replace(link_names=[prefix + name for name in sub_sys.link_names])

    if at_body is None:
        parent_sub_sys = -1
    else:
        parent_sub_sys = sys.name_to_idx(at_body)

    # append sub_sys at index end and replace sub_sys world with `at_body`
    N = sys.num_links()
    incr_repl = lambda p: p + N if p != -1 else parent_sub_sys
    sub_sys = sub_sys.replace(link_parents=[incr_repl(p) for p in sub_sys.link_parents])

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
