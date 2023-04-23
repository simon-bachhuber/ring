from typing import Tuple

import jax

from x_xy import algebra, base, scan
from x_xy.algorithms import jcalc


def forward_kinematics_transforms(
    sys: base.System, q: jax.Array
) -> Tuple[base.Transform, base.System]:
    """Perform forward kinematics in system.

    Returns:
        - Transforms from base to links. Transforms first axis is (n_links,).
        - Updated system object with updated `transform2` and `transform` fields.
    """

    p_to_l = {-1: base.Transform.zero()}

    def update_p_to_l(_, __, q, link, link_idx, parent_idx, joint_type: str):
        transform2 = jcalc.jcalc_transform(joint_type, q, link.joint_params)
        transform = algebra.transform_mul(transform2, link.transform1)
        link = link.replace(transform=transform, transform2=transform2)
        p_to_l_trafo = algebra.transform_mul(transform, p_to_l[parent_idx])
        p_to_l[link_idx] = p_to_l_trafo
        return p_to_l_trafo, link

    p_to_l_trafos, updated_links = scan.tree(
        sys,
        update_p_to_l,
        "qllll",
        q,
        sys.links,
        list(range(sys.num_links())),
        sys.link_parents,
        sys.link_types,
    )
    sys = sys.replace(links=updated_links)
    return (p_to_l_trafos, sys)


def forward_kinematics(
    sys: base.System, state: base.State
) -> Tuple[base.System, base.State]:
    """Perform forward kinematics in system.
    - Updates `transform` and `transform2` in `sys`
    - Updates `x` in `state`
    """
    x, sys = forward_kinematics_transforms(sys, state.q)
    state = state.replace(x=x)
    return sys, state
