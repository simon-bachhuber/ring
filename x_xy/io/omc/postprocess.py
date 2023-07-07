import logging
from typing import Optional

import jax
import jax.numpy as jnp
import tree_utils

import x_xy


def omc_to_xs(
    sys: x_xy.base.System,
    omc_data: dict,
    t1: float = 0.0,
    t2: Optional[float] = None,
    eps_frame: Optional[str] = None,
    qinv: bool = True,
) -> x_xy.base.Transform:
    """Build time-series of maximal coordinates `xs` using `omc_data`.

    Args:
        sys (x_xy.base.System): System which defines ordering of returned `xs`
        omc_data (dict): The optical motion capture data. Obtained using `process_omc`
        t1 (float, optional): Crop time left. Defaults to 0.0.
        t2 (Optional[float], optional): Crop time right. Defaults to None.
        eps_frame (str, optional): Move into this segment's frame at time zero as
            eps frame. Defaults to `None`.
            If `None`: Use root-frame as eps-frame.
            If 'none': Don't move into a specific eps-frame.

    Returns:
        x_xy.base.Transform: Time-series of eps-to-link transformations
    """

    if eps_frame == "none":
        logging.warning(
            "`eps_frame` set to `none` might lead to problems with artificial IMUs,"
            " since the gravity vector is assumed to be in positive z-axis in eps-frame"
        )

    # crop time left and right
    if t2 is None:
        t2i = tree_utils.tree_shape(omc_data)
    else:
        t2i = int(t2 / sys.dt)
    t1i = int(t1 / sys.dt)
    omc_data = jax.tree_map(lambda arr: jnp.array(arr)[t1i:t2i], omc_data)

    # determine `eps_frame` transform
    if eps_frame != "none":
        if eps_frame is None:
            connect_to_base = []
            # find link and link name that connects to world
            for link_name, link_parent in zip(sys.link_names, sys.link_parents):
                if link_parent == -1:
                    connect_to_base.append(link_name)
            assert len(connect_to_base) == 1, (
                f"Ambiguous `eps-frame` since multiple links ({connect_to_base})"
                " connect to base."
            )
            eps_frame = connect_to_base[0]
        eps = omc_data[eps_frame]
        q_eps = eps["quat"][0]
        if qinv:
            q_eps = x_xy.maths.quat_inv(q_eps)
        t_eps = x_xy.base.Transform(eps["pos"][0], q_eps)
    else:
        t_eps = x_xy.base.Transform.zero()

    # build `xs` from optical motion capture data
    xs = []

    def f(_, __, link_name: str):
        q, pos = omc_data[link_name]["quat"], omc_data[link_name]["pos"]
        if qinv:
            q = x_xy.maths.quat_inv(q)
        t = x_xy.base.Transform(pos, q)
        t = x_xy.algebra.transform_mul(t, x_xy.algebra.transform_inv(t_eps))
        xs.append(t)

    x_xy.scan.tree(sys, f, "l", sys.link_names)

    # stack and permute such that time-axis is 0-th axis
    xs = xs[0].batch(*xs[1:])
    xs = xs.transpose((1, 0, 2))
    return xs


def forward_kinematics_omc(
    sys: x_xy.base.System,
    xs: x_xy.base.Transform,
    delete_global_translation_rotation: bool = False,
    scale_revolute_joint_angles: Optional[float] = None,
) -> x_xy.base.Transform:
    """Perform forward kinematics. Use static transformations (transform1)
    from the `sys`, and extract and use joint transformations from `xs`.

    NOTE: The `sys` and `xs` must match in order and length.
    NOTE: For each link in the system the `rot` from `xs` is always used.
    The whole transform (including the field `pos`) is only used for
    links that connect to the worldbody.

    Args:
        sys (x_xy.base.System): System which defines scan order and `transform1`
        xs (x_xy.base.Transform): The optical motion capture data.
            Obtained using `process_omc` -> `omc_to_xs`
        delete_global_translation_rotation (bool): If set all transformations
            to the worldbody are unity.

    Returns:
        x_xy.base.Transform: Time-series of eps-to-link transformations
    """

    # consume time-axis of `xs`
    @jax.vmap
    def vmap_fk(xs):
        eps_to_l = {-1: x_xy.base.Transform.zero()}

        def fk(
            _,
            __,
            link_parent: int,
            link_idx: int,
            transform1: x_xy.base.Transform,
            link_type: str,
        ):
            if link_parent == -1:
                assert link_type == "free"
                if delete_global_translation_rotation:
                    transform2 = x_xy.base.Transform.zero()
                else:
                    transform2 = xs[link_idx]
            else:
                assert link_type not in ["px", "py", "pz", "free"]
                transform_opt = x_xy.algebra.transform_mul(
                    xs[link_idx], x_xy.algebra.transform_inv(xs[link_parent])
                )
                transform_opt_rot = transform_opt.rot
                if scale_revolute_joint_angles is not None:
                    axis, angle = x_xy.maths.quat_to_rot_axis(transform_opt_rot)
                    transform_opt_rot = x_xy.maths.quat_rot_axis(
                        axis, angle * scale_revolute_joint_angles
                    )

                transform2 = x_xy.base.Transform.create(rot=transform_opt_rot)

            transform = x_xy.algebra.transform_mul(transform2, transform1)
            eps_to_l[link_idx] = x_xy.algebra.transform_mul(
                transform, eps_to_l[link_parent]
            )
            return eps_to_l[link_idx]

        return x_xy.scan.tree(
            sys,
            fk,
            "llll",
            sys.link_parents,
            list(range(sys.num_links())),
            sys.links.transform1,
            sys.link_types,
        )

    return vmap_fk(xs)
