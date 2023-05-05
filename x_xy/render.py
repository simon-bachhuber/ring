import time
from abc import ABC, abstractmethod, abstractstaticmethod
from functools import partial
from pathlib import Path
from typing import Optional, TypeVar, Union

import imageio
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from tree_utils import tree_batch
from vispy import app, scene
from vispy.scene import MatrixTransform

from x_xy import algebra, base, maths
from x_xy.base import Box, Capsule, Cylinder, Geometry, Sphere

Camera = TypeVar("Camera")
Visual = TypeVar("Visual")
VisualPosOri = TypeVar("VisualPosOri")


class _AbstractRenderer(ABC):
    """
    Example:
        >> renderer = Renderer()
        >> renderer.init(sys.geoms)
        >> for x in xs:
        >>   renderer.update(x)
        >>   image = renderer.render()
    """

    @abstractmethod
    def _get_camera(self) -> Camera:
        pass

    @abstractmethod
    def _set_camera(self, camera: Camera) -> None:
        pass

    @abstractmethod
    def _render(self) -> jax.Array:
        pass

    def render(
        self, camera: Optional[Camera | list[Camera]] = None
    ) -> jax.Array | list[jax.Array]:
        if camera is None:
            camera = self._get_camera()

        if isinstance(camera, Camera):
            self._set_camera(camera)
            return self._render()

        images = []
        for cam in camera:
            self._set_camera(cam)
            images.append(self._render())
        return images

    @staticmethod
    def _add_box(geom: Box) -> Visual:
        raise NotImplementedError

    @staticmethod
    def _add_sphere(geom: Sphere) -> Visual:
        raise NotImplementedError

    @staticmethod
    def _add_cylinder(geom: Cylinder) -> Visual:
        raise NotImplementedError

    @staticmethod
    def _add_capsule(geom: Capsule) -> Visual:
        raise NotImplementedError

    def init(self, geoms: list[Geometry]):
        self.geoms = geoms
        self._fresh_init = True

        self.geom_link_idx = []
        self.geom_transform = []
        self.visuals = []
        for geom in geoms:
            self.geom_link_idx.append(geom.link_idx)
            self.geom_transform.append(geom.transform)
            if isinstance(geom, Box):
                visual = self._add_box(geom)
            elif isinstance(geom, Sphere):
                visual = self._add_sphere(geom)
            elif isinstance(geom, Cylinder):
                visual = self._add_cylinder(geom)
            elif isinstance(geom, Capsule):
                visual = self._add_capsule(geom)
            else:
                raise Exception(f"Unknown geom type: {type(geom)}")
            self.visuals.append(visual)

        self.geom_link_idx = tree_batch(self.geom_link_idx, backend="jax")
        self.geom_transform = tree_batch(self.geom_transform, backend="jax")

    @abstractstaticmethod
    def _compute_transform_per_visual(
        x_links: base.Transform,
        x_link_to_geom: base.Transform,
        geom_link_idx: jax.Array[int],
    ) -> VisualPosOri:
        "This can easily account for possible convention differences"
        pass

    @abstractmethod
    def _init_visual(self, visual: Visual, transform: VisualPosOri, geom: Geometry):
        pass

    def _update_visual(self, visual: Visual, transform: VisualPosOri, geom: Geometry):
        self._init_visual(visual, transform, geom)

    def update(self, x: base.Transform):
        "`x` are (n_links,) Transforms."

        # step 1: pre-compute all required transforms
        transform_per_visual = jax.jit(
            jax.vmap(self._compute_transform_per_visual, in_axes=(None, 0, 0))
        )(x, self.geom_transform, self.geom_link_idx)

        # step 2: update visuals
        for i, (visual, geom) in enumerate(zip(self.visuals, self.geoms)):
            t = transform_per_visual[i]
            if self._fresh_init:
                self._init_visual(visual, t, geom)
            else:
                self._update_visual(visual, t, geom)


@jax.jit
@partial(jax.vmap, in_axes=(None, 0, 0))
def _transform_4x4(x_links, geom_t, geom_link_idx):
    x = x_links[geom_link_idx]
    x = algebra.transform_mul(geom_t, x)
    E = maths.quat_to_3x3(x.rot)
    M = jnp.eye(4)
    M = M.at[:3, :3].set(E)
    T = jnp.eye(4)
    T = T.at[3, :3].set(x.pos)
    return M @ T


def transform_4x4(
    x_links: base.Transform, geom_transforms: base.Transform, geom_link_idxs: jax.Array
):
    return np.asarray(_transform_4x4(x_links, geom_transforms, geom_link_idxs))


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

        self.visuals = None
        self.geom_link_idx = None
        self.geom_transform = None
        self._populate(geoms)

    def _create_visual_element(self, geom: Geometry, **kwargs):
        if isinstance(geom, Box):
            return scene.visuals.Box(
                geom.dim_x, geom.dim_z, geom.dim_y, parent=self.view.scene, **kwargs
            )
        raise NotImplementedError()

    def _populate(self, geoms: list[list[Geometry]]):
        self._can_mutate = False
        self.visuals = []
        self.geom_link_idx = []
        self.geom_transform = []

        if self.show_cs:
            scene.visuals.XYZAxis(parent=self.view.scene)

        def append(visual, link_idx, transform):
            self.visuals.append(visual)
            self.geom_link_idx.append(link_idx)
            self.geom_transform.append(transform)

        for link_idx, geoms_per_link in enumerate(geoms):
            if self.show_cs:
                visual = scene.visuals.XYZAxis(parent=self.view.scene)
                append(visual, link_idx, base.Transform.zero())

            for geom in geoms_per_link:
                visual = self._create_visual_element(geom, **geom.vispy_kwargs)
                append(visual, link_idx, geom.transform)

        self.geom_link_idx = tree_batch(self.geom_link_idx, backend="jax")
        self.geom_transform = tree_batch(self.geom_transform, backend="jax")

    def change_camera(self, camera: scene.cameras.BaseCamera):
        "Change the camera angle of rendered image."
        self.view.camera = camera

    def update(self, x_links: base.Transform):
        "Update the link coordinates of the scene."
        self._x_links = x_links
        self._update_scene()
        self._can_mutate = True

    def render(self) -> np.ndarray:
        """Render scene. RGBA Array of Shape = (M, N, 4)"""
        return self.canvas.render(alpha=True)

    def _update_scene(self):
        # step 1: pre-compute all required 4x4 matrices (uses jax.vmap)
        t_4x4 = transform_4x4(self._x_links, self.geom_transform, self.geom_link_idx)

        # step 2: update visuals
        for i, visual in enumerate(self.visuals):
            if self._can_mutate:
                visual.transform.matrix = t_4x4[i]
            else:
                visual.transform = MatrixTransform(t_4x4[i])


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
    n_links = np.max(scene.geom_link_idx) + 1
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
