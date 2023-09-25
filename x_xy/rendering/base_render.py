from typing import Optional

import numpy as np
import tqdm

from .. import base
from ..utils import to_list

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
}


def render(
    sys: base.System,
    xs: base.Transform | list[base.Transform],
    camera: Optional[str] = None,
    show_pbar: bool = True,
    backend: str = "mujoco",
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
    if backend == "mujoco":
        from x_xy.rendering.mujoco_render import MujocoScene

        scene = MujocoScene(**scene_kwargs)
    elif backend == "vispy":
        from x_xy.rendering.vispy_render import VispyScene

        scene = VispyScene(**scene_kwargs)
    else:
        raise NotImplementedError

    # convert all colors to rgbas
    geoms_rgba = [_color_to_rgba(geom) for geom in sys.geoms]

    xs = to_list(xs)

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

    scene.init(geoms_rgba)

    frames = []
    for x in tqdm.tqdm(xs, "Rendering frames..", disable=not show_pbar):
        scene.update(x)
        frames.append(scene.render(camera=camera))

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
