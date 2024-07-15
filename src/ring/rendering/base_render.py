from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from ring import algebra
from ring import base
from ring import maths
from ring import sim2real
from ring import utils
from ring.algorithms import kinematics

_rgbas = {
    "self": (0.7, 0.5, 0.1, 1.0),
    "effector": (0.7, 0.4, 0.1, 1.0),
    "decoration": (0.3, 0.5, 0.7, 1.0),
    "eye": (0.0, 0.2, 1.0, 1.0),
    "target": (0.6, 0.3, 0.3, 1.0),
    "site": (0.5, 0.5, 0.5, 0.3),
    "red": (0.8, 0.2, 0.2, 1.0),
    "green": (0.2, 0.8, 0.2, 1.0),
    "blue": (0.2, 0.2, 0.8, 1.0),
    "yellow": (0.8, 0.8, 0.2, 1.0),
    "cyan": (0.2, 0.8, 0.8, 1.0),
    "magenta": (0.8, 0.2, 0.8, 1.0),
    "white": (0.8, 0.8, 0.8, 1.0),
    "gray": (0.5, 0.5, 0.5, 1.0),
    "brown": (0.6, 0.3, 0.1, 1.0),
    "orange": (0.8, 0.5, 0.2, 1.0),
    "pink": (0.8, 0.75, 0.8, 1.0),
    "purple": (0.5, 0.2, 0.5, 1.0),
    "lime": (0.5, 0.8, 0.2, 1.0),
    "gold": (0.8, 0.84, 0.2, 1.0),
    "matplotlib_green": (0.0, 0.502, 0.0, 1.0),
    "matplotlib_blue": (0.012, 0.263, 0.8745, 1.0),
    "matplotlib_lightblue": (0.482, 0.784, 0.9647, 1.0),
    "matplotlib_salmon": (0.98, 0.502, 0.447, 1.0),
    "black": (0.1, 0.1, 0.1, 1.0),
    "dustin_exp_blue": (75 / 255, 93 / 255, 208 / 255, 1.0),
    "dustin_exp_white": (241 / 255, 239 / 255, 208 / 255, 1.0),
    "dustin_exp_orange": (227 / 255, 139 / 255, 61 / 255, 1.0),
}


_args = None
_scene = None


def _load_scene(sys, backend, **scene_kwargs):
    global _args, _scene

    args = (sys, backend, scene_kwargs)
    if _args is not None:
        if utils.tree_equal(_args, args):
            return _scene

    _args = args
    if backend == "mujoco":
        utils.import_lib("mujoco")
        from ring.rendering.mujoco_render import MujocoScene

        scene = MujocoScene(**scene_kwargs)
    elif backend == "vispy":
        vispy = utils.import_lib("vispy")

        if "vispy_backend" in scene_kwargs:
            vispy_backend = scene_kwargs.pop("vispy_backend")
        else:
            vispy_backend = "pyqt6"

        vispy.use(vispy_backend)

        from ring.rendering.vispy_render import VispyScene

        scene = VispyScene(**scene_kwargs)
    else:
        raise NotImplementedError

    # mujoco does not implement the xyz Geometry; instead replace it with
    # three capsule Geometries
    geoms = sys.geoms
    if backend == "mujoco":
        geoms = _replace_xyz_geoms(geoms)

    # convert all colors to rgbas
    geoms_rgba = [_color_to_rgba(geom) for geom in geoms]

    scene.init(geoms_rgba)

    _scene = scene
    return _scene


def render(
    sys: base.System,
    xs: Optional[base.Transform | list[base.Transform]] = None,
    camera: Optional[str] = None,
    show_pbar: bool = True,
    backend: str = "mujoco",
    render_every_nth: int = 1,
    **scene_kwargs,
) -> list[np.ndarray]:
    """Render frames from system and trajectory of maximal coordinates `xs`.

    Args:
        sys (base.System): System to render.
        xs (base.Transform | list[base.Transform]): Single or time-series
        of maximal coordinates `xs`.
        show_pbar (bool, optional): Whether or not to show a progress bar.
        Defaults to True.

    Returns:
        list[np.ndarray]: Stacked rendered frames. Length == len(xs).
    """

    if xs is None:
        xs = kinematics.forward_kinematics(sys, base.State.create(sys))[1].x

    # convert time-axis of batched xs object into a list of unbatched x objects
    if isinstance(xs, base.Transform) and xs.ndim() == 3:
        xs = [xs[t] for t in range(xs.shape())]

    # ensure that a single unbatched x object is also a list
    xs = utils.to_list(xs)

    if render_every_nth != 1:
        xs = [xs[t] for t in range(0, len(xs), render_every_nth)]

    n_links = sys.num_links()

    def data_check(x):
        assert (
            x.pos.ndim == x.rot.ndim == 2
        ), f"Expected shape = (n_links, 3/4). Got pos.shape{x.pos.shape}, "
        "rot.shape={x.rot.shape}"
        assert (
            x.pos.shape[0] == x.rot.shape[0] == n_links
        ), "Number of links does not match"

    for x in xs:
        data_check(x)

    scene = _load_scene(sys, backend, **scene_kwargs)

    frames = []
    for x in tqdm.tqdm(xs, "Rendering frames..", disable=not show_pbar):
        scene.update(x)
        frames.append(scene.render(camera=camera))

    return frames


def _render_prediction_internals(
    sys, xs, yhat, transparent_segment_to_root, offset_truth, offset_pred
):
    if isinstance(xs, list):
        # list -> batched Transform
        xs = xs[0].batch(*xs[1:])

    sys_noimu, _ = sys.make_sys_noimu()

    if isinstance(yhat, (np.ndarray, jax.Array)):
        yhat = {name: yhat[..., i, :] for i, name in enumerate(sys_noimu.link_names)}

    xs_noimu = sim2real.match_xs(sys_noimu, xs, sys)

    # `yhat` are child-to-parent transforms, but we need parent-to-child
    # but not for those that connect to root, those are already parent-to-child
    transform2hat_rot = {}
    for name, p in zip(sys_noimu.link_names, sys_noimu.link_parents):
        if p == -1:
            transform2hat_rot[name] = yhat[name]
        else:
            transform2hat_rot[name] = maths.quat_inv(yhat[name])

    transform1, transform2 = sim2real.unzip_xs(sys_noimu, xs_noimu)

    # we add the missing links in transform2hat, links that connect to worldbody
    transform2hat = []
    for i, name in enumerate(sys_noimu.link_names):
        if name in transform2hat_rot:
            transform2_name = base.Transform.create(rot=transform2hat_rot[name])
        else:
            transform2_name = transform2.take(i, axis=1)
        transform2hat.append(transform2_name)

    # after transpose shape is (n_timesteps, n_links, ...)
    transform2hat = transform2hat[0].batch(*transform2hat[1:]).transpose((1, 0, 2))

    xshat = sim2real.zip_xs(sys_noimu, transform1, transform2hat)

    # swap time axis, and link axis
    xs, xshat = xs.transpose((1, 0, 2)), xshat.transpose((1, 0, 2))

    add_offset = lambda x, offset: algebra.transform_mul(
        x, base.Transform.create(pos=offset)
    )

    # create mapping from `name` -> Transform
    xs_dict = dict(
        zip(
            ["hat_" + name for name in sys_noimu.link_names],
            [add_offset(xshat[i], offset_pred) for i in range(sys_noimu.num_links())],
        )
    )
    xs_dict.update(
        dict(
            zip(
                sys.link_names,
                [add_offset(xs[i], offset_truth) for i in range(sys.num_links())],
            )
        )
    )

    sys_render = _sys_render(sys, transparent_segment_to_root)
    xs_render = []
    for name in sys_render.link_names:
        xs_render.append(xs_dict[name])
    xs_render = xs_render[0].batch(*xs_render[1:])
    xs_render = xs_render.transpose((1, 0, 2))

    return sys_render, xs_render


def render_prediction(
    sys: base.System,
    xs: base.Transform | list[base.Transform],
    yhat: dict | jax.Array | np.ndarray,
    # by default we don't predict the global rotation
    transparent_segment_to_root: bool = True,
    **kwargs,
):
    "`xs` matches `sys`. `yhat` matches `sys_noimu`. `yhat` are child-to-parent."

    offset_truth = jnp.array(kwargs.pop("offset_truth", [0.0, 0, 0]))
    offset_pred = jnp.array(kwargs.pop("offset_pred", [0.0, 0, 0]))

    sys_render, xs_render = jax.jit(_render_prediction_internals, static_argnums=3)(
        sys, xs, yhat, transparent_segment_to_root, offset_truth, offset_pred
    )

    frames = render(sys_render, xs_render, **kwargs)
    return frames


def _color_to_rgba(geom: base.Geometry) -> base.Geometry:
    if geom.color is None:
        new_color = _rgbas["self"]
    elif isinstance(geom.color, tuple):
        if len(geom.color) == 3:
            new_color = geom.color + (1.0,)
        else:
            new_color = geom.color
    elif isinstance(geom.color, str):
        new_color = _rgbas[geom.color]
    else:
        raise NotImplementedError

    return geom.replace(color=new_color)


def _xyz_to_three_capsules(xyz: base.XYZ) -> list[base.Geometry]:
    capsules = []
    length = xyz.size
    radius = length / 7
    colors = ["red", "green", "blue"]
    rot_axis = [1, 0, 2]

    for i, (color, axis) in enumerate(zip(colors, rot_axis)):
        pos = maths.unit_vectors(i) * length / 2
        rot = maths.quat_rot_axis(maths.unit_vectors(axis), jnp.pi / 2)
        t = algebra.transform_mul(base.Transform(pos, rot), xyz.transform)
        capsules.append(
            base.Capsule(0.0, t, xyz.link_idx, color, xyz.edge_color, radius, length)
        )
    return capsules


def _replace_xyz_geoms(geoms: list[base.Geometry]) -> list[base.Geometry]:
    geoms_replaced = []
    for geom in geoms:
        if isinstance(geom, base.XYZ):
            geoms_replaced += _xyz_to_three_capsules(geom)
        else:
            geoms_replaced.append(geom)
    return geoms_replaced


def _sys_render(
    sys: base.Transform, transparent_segment_to_root: bool
) -> base.Transform:
    sys_noimu, _ = sys.make_sys_noimu()

    def _geoms_replace_color(sys: base.System, color):
        keep = lambda i: (not transparent_segment_to_root) or sys.link_parents[i] != -1
        geoms = [g.replace(color=color) for g in sys.geoms if keep(g.link_idx)]
        return sys.replace(geoms=geoms)

    # replace render color of geoms for render of predicted motion
    prediction_color = (78 / 255, 163 / 255, 243 / 255, 1.0)
    sys_newcolor = _geoms_replace_color(sys_noimu, prediction_color)
    sys_render = sys.inject_system(sys_newcolor.add_prefix_suffix("hat_"))

    return sys_render
