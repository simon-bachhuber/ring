import time
from pathlib import Path
from typing import Optional, Union

import imageio
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from vispy import app, scene
from vispy.scene import MatrixTransform

from x_xy import base, maths
from x_xy.base import Box, Geometry


@jax.jit
def _transform_4x4(pos, rot, com):
    E = maths.quat_to_3x3(rot)
    M = jnp.eye(4)
    M = M.at[:3, :3].set(E)
    pos = pos + E.T @ com
    T = jnp.eye(4)
    T = T.at[3, :3].set(pos)
    return M @ T


def transform_4x4(pos, rot, com=jnp.zeros((3,))):
    return np.asarray(_transform_4x4(pos, rot, com))


def _enable_headless_backend():
    import vispy

    try:
        vispy.use("egl")
        return True
    except RuntimeError:
        try:
            vispy.use("osmesa")
            return True
        except RuntimeError:
            print(
                "Headless mode requires either `egl` or `osmesa` as backends for vispy",
                "Couldn't find neither. Falling back to interactive mode.",
            )
            return False


class VispyScene:
    def __init__(
        self,
        geoms: list[list[Geometry]],
        show_cs=True,
        size=(1280, 720),
        camera: scene.cameras.BaseCamera = scene.TurntableCamera(
            elevation=30, distance=6
        ),
        headless: bool = False,
        **kwargs,
    ):
        """Scene which can be rendered.

        Args:
            geoms (list[list[Geometry]]): A list of list of geometries per link.
                len(geoms) == number of links in system
            show_cs (bool, optional): Show coordinate system of links.
                Defaults to True.
            size (tuple, optional): Width and height of rendered image.
                Defaults to (1280, 720).
            camera (scene.cameras.BaseCamera, optional): The camera angle.
                Defaults to scene.TurntableCamera( elevation=30, distance=6 ).
            headless (bool, optional): Headless if the worker can not open windows.
                Defaults to False.

        Example:
            >> scene = VispyScene(sys.geoms)
            >> scene.update(state.x)
            >> image = scene.render()
        """
        self.headless = False
        if headless:
            # returns `True` if successfully found backend
            self.headless = _enable_headless_backend()

        self.canvas = scene.SceneCanvas(
            keys="interactive", size=size, show=True, **kwargs
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = camera
        self.show_cs = show_cs
        self.geoms = geoms
        self._populate()

    def _create_visual_element(self, geom: Geometry, **kwargs):
        if isinstance(geom, Box):
            return scene.visuals.Box(
                geom.dim_x, geom.dim_z, geom.dim_y, parent=self.view.scene, **kwargs
            )
        raise NotImplementedError()

    def _populate(self):
        self._can_mutate = False
        self.visuals = []
        self._geoms_cs = []

        if self.show_cs:
            scene.visuals.XYZAxis(parent=self.view.scene)

        for geoms_per_link in self.geoms:
            if self.show_cs:
                self._geoms_cs.append(scene.visuals.XYZAxis(parent=self.view.scene))

            visuals_per_link = []
            for geom in geoms_per_link:
                visuals_per_link.append(
                    self._create_visual_element(geom, **geom.vispy_kwargs)
                )
            self.visuals.append(visuals_per_link)

    def change_camera(self, camera: scene.cameras.BaseCamera):
        "Change the camera angle of rendered image."
        self.view.camera = camera

    def update(self, x: base.Transform):
        "Update the link coordinates of the scene."
        self.data_pos = x.pos
        self.data_rot = x.rot
        self._update_scene()
        self._can_mutate = True

    def render(self) -> np.ndarray:
        """Render scene. RGBA Array of Shape = (M, N, 4)"""
        return self.canvas.render(alpha=True)

    def _get_link_data(self, link_idx: int):
        rot = self.data_rot[link_idx]
        pos = self.data_pos[link_idx]
        return rot, pos

    def _get_transform_matrix(self, link_idx, geom_idx=None):
        rot, pos = self._get_link_data(link_idx)

        if geom_idx is not None:
            return transform_4x4(pos, rot, self.geoms[link_idx][geom_idx].CoM)
        else:
            return transform_4x4(pos, rot)

    def _update_scene(self):
        for link_idx in range(len(self.visuals)):
            if self.show_cs:
                transform_matrix = self._get_transform_matrix(link_idx)
                cs = self._geoms_cs[link_idx]
                if self._can_mutate:
                    cs.transform.matrix = transform_matrix
                else:
                    cs.transform = MatrixTransform(transform_matrix)

            for geom_idx in range(len(self.visuals[link_idx])):
                transform_matrix = self._get_transform_matrix(link_idx, geom_idx)
                visual = self.visuals[link_idx][geom_idx]
                if self._can_mutate:
                    visual.transform.matrix = transform_matrix
                else:
                    visual.transform = MatrixTransform(transform_matrix)


def _parse_timestep(timestep: float, fps: int, N: int):
    assert 1 / timestep > fps, "The `fps` is too high for the simulated timestep"
    fps_simu = int(1 / timestep)
    assert (fps_simu % fps) == 0, "The `fps` does not align with the timestep"
    T = N * timestep
    step = int(fps_simu / fps)
    return T, step


def _data_checks(scene, data_pos, data_rot):
    assert (
        data_pos.ndim == data_rot.ndim == 3
    ), "Expected shape = (n_timesteps, n_links, 3/4)"
    n_links = len(scene.geoms)
    assert (
        data_pos.shape[1] == data_rot.shape[1] == n_links
    ), "Number of links does not match"


def _infer_extension_from_path(path: Path) -> Optional[str]:
    ext = path.suffix
    # fmt starts after the . e.g. .mp4
    return ext[1:] if len(ext) > 0 else None


def animate(
    path: Union[str, Path],
    scene: VispyScene,
    x: base.Transform,
    dt: float,
    fps: int = 50,
    fmt: str = "mp4",
):
    """Make animation from scene and trajectory of maximal coordinates. `x`
    are stacked in time along 0th-axis.
    """
    path = Path(path)
    file_fmt = _infer_extension_from_path(path)

    if file_fmt is not None:
        assert (
            file_fmt == fmt.lower()
        ), f"""The chosen filename `{path.name}` and required fmt `{fmt}`
        are inconsistent."""

    if file_fmt is None:
        path = path.with_suffix("." + fmt)

    # assert fmt.lower() == "gif", "Currently the only implemented option is `GIF`"

    if not scene.headless:
        print(
            """Warning: Animate function expected a `Renderer(headless=True)`.
            This way we don't open any GUI windows."""
        )

    _data_checks(scene, x.pos, x.rot)

    N = x.pos.shape[0]
    _, step = _parse_timestep(dt, fps, N)

    frames = []
    for t in tqdm.tqdm(range(0, N, step), "Rendering frames.."):
        scene.update(x[t])
        frames.append(scene.render())

    print(f"DONE. Converting frames to {path} (this might take a while..)")
    # duration = int(1000 * 1 / fps)
    imageio.mimsave(path, frames, format=fmt, fps=fps)


class Window:
    def __init__(
        self,
        scene: VispyScene,
        x: base.Transform,
        timestep: float,
        fps: int = 50,
    ) -> None:
        """Open an interactive Window that plays back the pre-computed trajectory.

        Args:
            scene (VispyScene): Scene used for rendering.
            x (base.Transform): Pre-computed trajectory.
            timestep (float): Timedelta between Transforms.
            fps (int, optional): Frame-rate. Defaults to 50.
        """
        _data_checks(scene, x.pos, x.rot)
        self._x = x
        self._scene = scene

        self.N = x.pos.shape[0]
        self.T, self.step = _parse_timestep(timestep, fps, self.N)
        self.timestep = timestep
        self.fps = fps

    def reset(self):
        "Reset trajectory to beginning."
        self.reached_end = False
        self.time = 0
        self.t = 0
        self.starttime = time.time()
        self._update_scene()

    def _update_scene(self):
        self._scene.update(self._x[self.t])

    def _on_timer(self, event):
        if self.time > self.T:
            self.reached_end = True

        if self.reached_end:
            return

        self._update_scene()

        self.t += self.step
        self.time += self.step * self.timestep
        self.realtime = time.time()
        self.current_fps = (self.time / (self.realtime - self.starttime)) * self.fps

        print("FPS: ", int(self.current_fps), f"Target FPS: {self.fps}")

    def open(self):
        "Open interactive GUI window."
        self.reset()

        self._timer = app.Timer(
            1 / self.fps,
            connect=self._on_timer,
            start=True,
        )

        app.run()
