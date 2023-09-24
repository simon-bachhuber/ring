import numpy as np
import tqdm

from .. import base
from ..utils import to_list


def render(
    sys: base.System,
    xs: base.Transform | list[base.Transform],
    show_pbar: bool = True,
    backend: str = "mujoco",
    **kwargs,
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

        scene = MujocoScene(**kwargs)
    elif backend == "vispy":
        from x_xy.rendering.vispy_render import VispyScene

        scene = VispyScene(**kwargs)
    else:
        raise NotImplementedError

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

    scene.init(sys.geoms)

    frames = []
    for x in tqdm.tqdm(xs, "Rendering frames..", disable=not show_pbar):
        scene.update(x)
        frames.append(scene.render())

    return frames
