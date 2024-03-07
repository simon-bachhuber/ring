from typing import Optional

import jax
import jax.numpy as jnp
from ring import base
from tree_utils import tree_batch


def _tree_nan_like(tree, repeats: int):
    return jax.tree_map(
        lambda arr: jnp.repeat(arr[0:1] * jnp.nan, repeats, axis=0), tree
    )


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

    # build union of two joint_params dictionaries because each system might have custom
    # joints that the other does not have
    missing_in_sys = set(sub_sys.links.joint_params.keys()) - set(
        sys.links.joint_params.keys()
    )
    sys_n_links = sys.num_links()
    for typ in missing_in_sys:
        sys.links.joint_params[typ] = _tree_nan_like(
            sub_sys.links.joint_params[typ], sys_n_links
        )

    missing_in_subsys = set(
        sys.links.joint_params.keys() - sub_sys.links.joint_params.keys()
    )
    subsys_n_links = sub_sys.num_links()
    for typ in missing_in_subsys:
        sub_sys.links.joint_params[typ] = _tree_nan_like(
            sys.links.joint_params[typ], subsys_n_links
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
        omc=sys.omc + sub_sys.omc,
    )

    return combined_sys.parse()
