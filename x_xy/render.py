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
from tree_utils import PyTree, tree_batch
from vispy import app, scene
from vispy.scene import MatrixTransform

from x_xy import algebra, base, maths
from x_xy.base import Box, Capsule, Cylinder, Geometry, Sphere

Camera = TypeVar("Camera")
Visual = TypeVar("Visual")
VisualPosOri1 = PyTree
VisualPosOri2 = PyTree


class Scene(ABC):
    _xyz: bool = True

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

    def enable_xyz(self) -> None:
        self._xyz = True

    def disable_xyz(self) -> None:
        self._xyz = False

    def render(
        self, camera: Optional[Camera | list[Camera]] = None
    ) -> jax.Array | list[jax.Array]:
        "Returns: RGBA Array of Shape = (M, N, 4)"
        if camera is None:
            camera = self._get_camera()

        if not isinstance(camera, list):
            self._set_camera(camera)
            return self._render()

        images = []
        for cam in camera:
            self._set_camera(cam)
            images.append(self._render())
        return images

    def _add_box(self, box: Box) -> Visual:
        return scene.visuals.Box(
                    box.dim_x, box.dim_z, box.dim_y, parent=self.view.scene, **kwargs
                )

    def _add_sphere(self, sphere: Sphere) -> Visual:
        return scene.visuals.Sphere(sphere.radius, parent=self.view.scene, **kwargs)

    def _add_cylinder(self, cyl: Cylinder) -> Visual:
        n_points = int(cyl.length * 100)

        tube_points = jnp.zeros((n_points, 3))
        tube_points = tube_points.at[:, 0].set(
            jnp.linspace(-cyl.length / 2, cyl.length / 2, n_points)
        )

        return scene.visuals.Tube(
            tube_points,
            jnp.full((n_points,), cyl.radius),
            closed=True,
            parent=self.view.scene,
            **kwargs,
        )

    def _add_capsule(self, geom: Capsule) -> Visual:
        raise NotImplementedError

    def _add_xyz(self) -> Visual:
        raise NotImplementedError

    def init(self, geoms: list[Geometry]):
        self.geoms = [geom for geom in geoms]
        self._fresh_init = True

        geom_link_idx = []
        geom_transform = []
        self.visuals = []
        for geom in geoms:
            geom_link_idx.append(geom.link_idx)
            geom_transform.append(geom.transform)
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

        if self._xyz:
            unique_link_indices = set(geom_link_idx)
            for unique_link_idx in unique_link_indices:
                geom_link_idx.append(unique_link_idx)
                geom_transform.append(base.Transform.zero())
                self.visuals.append(self._add_xyz())
                # otherwise the .update function won't iterate
                # over all visuals since it uses a zip(...)
                self.geoms.append(None)

            # add one final for root frame
            self._add_xyz()

        self.geom_link_idx = tree_batch(geom_link_idx, backend="jax")
        self.geom_transform = tree_batch(geom_transform, backend="jax")

    @abstractstaticmethod
    def _compute_transform_per_visual(
        x_links: base.Transform,
        x_link_to_geom: base.Transform,
        geom_link_idx: int,
    ) -> VisualPosOri1:
        "This can easily account for possible convention differences"
        pass

    @abstractstaticmethod
    def _postprocess_transforms(transform: VisualPosOri1) -> VisualPosOri2:
        pass

    @abstractmethod
    def _init_visual(
        self, visual: Visual, transform: VisualPosOri2, geom: None | Geometry
    ):
        pass

    def _update_visual(
        self, visual: Visual, transform: VisualPosOri2, geom: None | Geometry
    ):
        self._init_visual(visual, transform, geom)

    def update(self, x: base.Transform):
        "`x` are (n_links,) Transforms."

        # step 1: pre-compute all required transforms
        transform_per_visual = _compile_staticmethod(
            self._compute_transform_per_visual,
            x,
            self.geom_transform,
            self.geom_link_idx,
        )

        # step 2: postprocess all transforms once
        transform_per_visual = self._postprocess_transforms(transform_per_visual)

        # step 3: update visuals
        for i, (visual, geom) in enumerate(zip(self.visuals, self.geoms)):
            t = jax.tree_map(lambda arr: arr[i], transform_per_visual)
            if self._fresh_init:
                self._init_visual(visual, t, geom)
            else:
                self._update_visual(visual, t, geom)

        # step 4: unset flag
        self._fresh_init = False


@partial(jax.jit, static_argnums=0)
def _compile_staticmethod(static_method, x, geom_transform, geom_link_idx):
    return jax.vmap(static_method, in_axes=(None, 0, 0))(
        x, geom_transform, geom_link_idx
    )


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


class VispyScene(Scene):
    def __init__(
        self,
        show_cs=True,
        size=(1280, 720),
        camera: scene.cameras.BaseCamera = scene.TurntableCamera(
            elevation=30, distance=6
        ),
        headless_backend: bool = False,
        vispy_backend: Optional[str] = None,
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
            headless_backend (bool, optional): Headless if the worker can not open windows.
                Defaults to False.

        Example:
            >> scene = VispyScene()
            >> scene.init(sys.geoms)
            >> scene.update(state.x)
            >> image = scene.render()
        """
        if headless_backend:
            assert (
                vispy_backend is None
            ), "Can only set one backend. Either provide `vispy_backend` or enable `headless_backend`"

        self.headless = False
        if headless_backend:
            # returns `True` if successfully found backend
            self.headless = _enable_headless_backend()
        if vispy_backend is not None:
            import vispy

            vispy.use(vispy_backend)

        self.canvas = scene.SceneCanvas(
            keys="interactive", size=size, show=True, **kwargs
        )
        self.view = self.canvas.central_widget.add_view()
        self._set_camera(camera)
        if show_cs:
            self.enable_xyz()
        else:
            self.disable_xyz()

    def _create_visual_element(self, geom: Geometry, **kwargs):
        if isinstance(geom, Box):
            return scene.visuals.Box(
                geom.dim_x, geom.dim_z, geom.dim_y, parent=self.view.scene, **kwargs
            )

    def _set_camera(self, camera: scene.cameras.BaseCamera) -> None:
        self.view.camera = camera

    def _render(self) -> jax.Array:
        return self.canvas.render(alpha=True)

    def _add_box(self, geom: base.Box):
        return scene.visuals.Box(
            geom.dim_x,
            geom.dim_z,
            geom.dim_y,
            parent=self.view.scene,
            **geom.vispy_kwargs,
        )

    def _add_xyz(self) -> Visual:
        return scene.visuals.XYZAxis(parent=self.view.scene)

    @staticmethod
    def _compute_transform_per_visual(
        x_links: base.Transform,
        x_link_to_geom: base.Transform,
        geom_link_idx: int,
    ) -> jax.Array:
        x = jax.lax.cond(
            geom_link_idx == -1,
            lambda: base.Transform.zero(),
            lambda: x_links[geom_link_idx],
        )
        x = algebra.transform_mul(x_link_to_geom, x)
        E = maths.quat_to_3x3(x.rot)
        M = jnp.eye(4)
        M = M.at[:3, :3].set(E)
        T = jnp.eye(4)
        T = T.at[3, :3].set(x.pos)
        return M @ T

    @staticmethod
    def _postprocess_transforms(transform: jax.Array) -> np.ndarray:
        return np.asarray(transform)

    def _init_visual(
        self, visual: scene.visuals.VisualNode, transform: np.ndarray, geom: Geometry
    ):
        visual.transform = MatrixTransform(transform)

    def _update_visual(
        self, visual: scene.visuals.VisualNode, transform: np.ndarray, geom: Geometry
    ):
        visual.transform.matrix = transform


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


SCENE_BACKENDS = {
    "vispy": VispyScene,
}


def _make_scene(sys: base.System, backend: str, **backend_kwargs) -> Scene:
    scene = SCENE_BACKENDS[backend](**backend_kwargs)
    scene.init(sys.geoms)
    return scene


def animate(
    path: Union[str, Path],
    sys: base.System,
    x: base.Transform,
    fps: int = 50,
    fmt: str = "mp4",
    backend: str = "vispy",
    **backend_kwargs,
):
    """Make animation from system and trajectory of maximal coordinates. `x`
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

    scene = _make_scene(sys, backend, **backend_kwargs)
    _data_checks(scene, x.pos, x.rot)

    N = x.pos.shape[0]
    _, step = _parse_timestep(sys.dt, fps, N)

    frames = []
    for t in tqdm.tqdm(range(0, N, step), "Rendering frames.."):
        scene.update(x[t])
        frames.append(scene.render())

    print(f"DONE. Converting frames to {path} (this might take a while..)")
    imageio.mimsave(path, frames, format=fmt, fps=fps)


class Window:
    def __init__(
        self,
        sys: base.System,
        x: base.Transform,
        fps: int = 50,
        backend: str = "vispy",
        **backend_kwargs,
    ):
        """Open an interactive Window that plays back the pre-computed trajectory.

        Args:
            scene (VispyScene): Scene used for rendering.
            x (base.Transform): Pre-computed trajectory.
            timestep (float): Timedelta between Transforms.
            fps (int, optional): Frame-rate. Defaults to 50.
        """
        self._x = x
        self._scene = _make_scene(sys, backend, **backend_kwargs)
        _data_checks(self._scene, x.pos, x.rot)

        self.N = x.pos.shape[0]
        self.T, self.step = _parse_timestep(sys.dt, fps, self.N)
        self.timestep = sys.dt
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
