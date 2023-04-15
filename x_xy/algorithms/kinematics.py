from typing import Tuple

from x_xy import algebra, base, scan
from x_xy.algorithms import jcalc


def forward_kinematics_transforms(
    sys: base.System, q: dict
) -> Tuple[base.Transform, base.System]:
    """Perform forward kinematics in system.

    Returns:
        - Transforms from base to links. Transforms first axis is (n_links,).
        - Updated system object with updated `transform2` and `transform` fields.
    """

    def parent_to_link_transform(y, q, link: base.Link, joint_type: str):
        base_to_parent_transform, _ = y
        transform2 = jcalc.jcalc_transform(joint_type, q, link.joint_params)
        transform = algebra.transform_mul(transform2, link.transform1)
        link = link.replace(transform=transform, transform2=transform2)
        base_to_link_transform = algebra.transform_mul(
            transform, base_to_parent_transform
        )
        return base_to_link_transform, link

    dummy_link = sys.links.take(0)
    y0 = (base.Transform.zero(), dummy_link)
    base_to_link_transforms, new_links = scan.scan_links(
        sys, parent_to_link_transform, y0, q, sys.links, sys.link_joint_types
    )
    sys = sys.replace(links=new_links)
    return (base_to_link_transforms, sys)
