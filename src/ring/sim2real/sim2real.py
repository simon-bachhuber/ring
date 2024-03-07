from typing import Optional, Tuple

import jax
from ring import algebra
from ring import base
from ring import io
from ring import maths
from ring.algorithms import generator
from ring.algorithms import jcalc
import tree_utils


def xs_from_raw(
    sys: base.System,
    link_name_pos_rot: dict,
    eps_frame: Optional[str] = None,
    qinv: bool = False,
) -> base.Transform:
    """Build time-series of maximal coordinates `xs` from raw position and
    quaternion trajectory data. This function scans through each link (as
    defined by `sys`), looks for the raw data in `link_name_pos_rot` using
    the `link_name` as identifier. It inverts the quaternion if `qinv`.
    Then, it creates a `Transform` that transforms from epsilon (as defined
    by `eps_frame`) to the link for each timestep. Finally, it stacks all
    transforms in order as defined by `sys` along the 1-th axis. The 0-th
    axis is time axis.

    Args:
        sys (ring.base.System): System which defines ordering of returned `xs`
        link_name_pos_rot (dict): Dictonary of `link_name` ->
            {'pos': ..., 'quat': ...}. Obtained, e.g., using `process_omc`.
        eps_frame (str, optional): Move into this segment's frame at time zero as
            eps frame. Defaults to `None`.
            If `None`: Don't move into a specific eps-frame.

    Returns:
        ring.base.Transform: Time-series of eps-to-link transformations
    """
    # determine `eps_frame` transform
    if eps_frame is not None:
        eps = link_name_pos_rot[eps_frame]
        q_eps = eps["quat"][0]
        if qinv:
            q_eps = maths.quat_inv(q_eps)
        t_eps = base.Transform(eps["pos"][0], q_eps)
    else:
        t_eps = base.Transform.zero()

    # build `xs` from optical motion capture data
    xs = []

    def f(_, __, link_name: str):
        q = link_name_pos_rot[link_name]["quat"]
        pos = link_name_pos_rot[link_name].get("pos", None)
        if qinv:
            q = maths.quat_inv(q)
        t = base.Transform.create(pos, q)
        t = algebra.transform_mul(t, algebra.transform_inv(t_eps))
        xs.append(t)

    sys.scan(f, "l", sys.link_names)

    # stack and permute such that time-axis is 0-th axis
    xs = xs[0].batch(*xs[1:])
    xs = xs.transpose((1, 0, 2))
    return xs


def match_xs(
    sys: base.System, xs: base.Transform, sys_xs: base.System
) -> base.Transform:
    """Match tranforms `xs` to subsystem `sys`.

    Args:
        sys (System): Smaller system. Every link in `sys` must be in `sys_xs`.
        xs (Transform): Transforms of larger system.
        sys_xs (Transform): Larger system.

    Returns:
        Transform: Transforms of smaller system.
    """
    _checks_time_series_of_xs(sys_xs, xs)

    xs_small = xs_from_raw(
        sys,
        {
            name: {
                "pos": xs.pos[:, sys_xs.name_to_idx(name)],
                "quat": xs.rot[:, sys_xs.name_to_idx(name)],
            }
            for name in sys_xs.link_names
        },
        eps_frame=None,
        qinv=False,
    )
    return xs_small


def unzip_xs(
    sys: base.System, xs: base.Transform
) -> Tuple[base.Transform, base.Transform]:
    """Split eps-to-link transforms into parent-to-child pure
    translational `transform1` and pure rotational `transform2`.

    Args:
        sys (System): Defines scan.tree
        xs (Transform): Eps-to-link transforms

    Returns:
        Tuple[Transform, Transform]: transform1, transform2
    """
    _checks_time_series_of_xs(sys, xs)

    @jax.vmap
    def _unzip_xs(xs):
        def f(_, __, i: int, p: int):
            if p == -1:
                x_parent_to_link = xs[i]
            else:
                x_parent_to_link = algebra.transform_mul(
                    xs[i], algebra.transform_inv(xs[p])
                )

            transform1_pos = base.Transform.create(pos=x_parent_to_link.pos)
            transform2_rot = base.Transform.create(rot=x_parent_to_link.rot)
            return (transform1_pos, transform2_rot)

        return sys.scan(f, "ll", list(range(sys.num_links())), sys.link_parents)

    return _unzip_xs(xs)


def zip_xs(
    sys: base.System,
    xs_transform1: base.Transform,
    xs_transform2: base.Transform,
) -> base.Transform:
    """Performs forward kinematics using `transform1` and `transform2`.

    Args:
        sys (ring.base.System): Defines scan_sys
        xs_transform1 (ring.base.Transform): Applied before `transform1`
        xs_transform2 (ring.base.Transform): Applied after `transform2`

    Returns:
        ring.base.Transform: Time-series of eps-to-link transformations
    """
    _checks_time_series_of_xs(sys, xs_transform1)
    _checks_time_series_of_xs(sys, xs_transform2)

    @jax.vmap
    def _zip_xs(xs_transform1, xs_transform2):
        eps_to_l = {-1: base.Transform.zero()}

        def f(_, __, i: int, p: int):
            transform = algebra.transform_mul(xs_transform2[i], xs_transform1[i])
            eps_to_l[i] = algebra.transform_mul(transform, eps_to_l[p])
            return eps_to_l[i]

        return sys.scan(f, "ll", list(range(sys.num_links())), sys.link_parents)

    return _zip_xs(xs_transform1, xs_transform2)


def _checks_time_series_of_xs(sys, xs):
    assert tree_utils.tree_ndim(xs) == 3, f"pos.shape={xs.pos.shape}"
    num_links_xs, num_links_sys = tree_utils.tree_shape(xs, axis=1), sys.num_links()
    assert num_links_xs == num_links_sys, f"{num_links_xs} != {num_links_sys}"


def delete_to_world_pos_rot(sys: base.System, xs: base.Transform) -> base.Transform:
    """Replace the transforms of all links that connect to the worldbody
    by unity transforms.

    Args:
        sys (System): System only used for structure (in scan_sys).
        xs (Transform): Time-series of transforms to be modified.

    Returns:
        Transform: Time-series of modified transforms.
    """
    _checks_time_series_of_xs(sys, xs)

    zero_trafo = base.Transform.zero((xs.shape(),))
    for i, p in enumerate(sys.link_parents):
        if p == -1:
            xs = _overwrite_transform_of_link_then_update(sys, xs, zero_trafo, i)
    return xs


def randomize_to_world_pos_rot(
    key: jax.Array, sys: base.System, xs: base.Transform, config: jcalc.MotionConfig
) -> base.Transform:
    """Replace the transforms of all links that connect to the worldbody
    by randomize transforms.

    Args:
        key (jax.Array): PRNG Key.
        sys (System): System only used for structure (in scan_sys).
        xs (Transform): Time-series of transforms to be modified.
        config (MotionConfig): Defines the randomization.

    Returns:
        Transform: Time-series of modified transforms.
    """
    _checks_time_series_of_xs(sys, xs)
    assert sys.link_parents.count(-1) == 1, "Found multiple connections to world"

    free_sys_str = """
<x_xy>
    <options dt="0.01"/>
    <worldbody>
        <body name="free" joint="free"/>
    </worldbody>
</x_xy>
"""

    free_sys = io.load_sys_from_str(free_sys_str)
    _, xs_free = generator.RCMG(
        free_sys, config, finalize_fn=lambda key, q, x, sys: (q, x)
    ).to_lazy_gen()(key)
    xs_free = xs_free.take(0, axis=0)
    xs_free = xs_free.take(free_sys.name_to_idx("free"), axis=1)
    link_idx_to_world = sys.link_parents.index(-1)
    return _overwrite_transform_of_link_then_update(sys, xs, xs_free, link_idx_to_world)


def _overwrite_transform_of_link_then_update(
    sys: base.System, xs: base.Transform, xs_new_link: base.Transform, new_link_idx: int
):
    """Replace transform and then perform forward kinematics."""
    assert xs_new_link.ndim() == (xs.ndim() - 1) == 2
    transform1, transform2 = unzip_xs(sys, xs)
    transform1 = _replace_transform_of_link(transform1, xs_new_link, new_link_idx)
    zero_trafo = base.Transform.zero((xs_new_link.shape(),))
    transform2 = _replace_transform_of_link(transform2, zero_trafo, new_link_idx)
    return zip_xs(sys, transform1, transform2)


def _replace_transform_of_link(
    xs: base.Transform, xs_new_link: base.Transform, link_idx
):
    return xs.transpose((1, 0, 2)).index_set(link_idx, xs_new_link).transpose((1, 0, 2))


def scale_xs(
    sys: base.System,
    xs: base.Transform,
    factor: float,
    exclude: list[str] = ["px", "py", "pz", "free"],
) -> base.Transform:
    """Increase / decrease transforms by scaling their positional / rotational
    components based on the systems link type, i.e. the `xs` should conceptionally
    be `transform2` objects.

    Args:
        sys (System): System defining structure (for scan_sys)
        xs (Transform): Time-series of transforms to be modified.
        factor (float): Multiplicative factor.
        exclude (list[str], optional): Skip scaling of transforms if their link_type
            is one of those. Defaults to ["px", "py", "pz", "free"].

    Returns:
        Transform: Time-series of scaled transforms.
    """
    _checks_time_series_of_xs(sys, xs)

    @jax.vmap
    def _scale_xs(xs):
        def f(_, __, i: int, type: str):
            x_link = xs[i]
            if type not in exclude:
                x_link = _scale_transform_based_on_type(x_link, type, factor)
            return x_link

        return sys.scan(f, "ll", list(range(sys.num_links())), sys.link_types)

    return _scale_xs(xs)


def _scale_transform_based_on_type(x: base.Transform, link_type: str, factor: float):
    pos, rot = x.pos, x.rot
    if link_type in ["px", "py", "pz", "free"]:
        pos = pos * factor
    if link_type in ["rx", "ry", "rz", "spherical", "free"]:
        axis, angle = maths.quat_to_rot_axis(rot)
        rot = maths.quat_rot_axis(axis, angle * factor)
    return base.Transform(pos, rot)
