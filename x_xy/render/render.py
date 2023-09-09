from abc import ABC
from abc import abstractmethod
from abc import abstractstaticmethod
from functools import partial
from pathlib import Path
import time
from typing import Optional, Sequence, TypeVar, Union

import imageio
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import tree_utils
from tree_utils import PyTree
from tree_utils import tree_batch
from vispy import app
from vispy import scene
from vispy.scene import MatrixTransform

import x_xy
from x_xy import algebra
from x_xy import maths

from . import visuals
from .. import base
from ..base import Box
from ..base import Capsule
from ..base import Cylinder
from ..base import Geometry
from ..base import Sphere
from ..base import XYZ

Camera = TypeVar("Camera")
Visual = TypeVar("Visual")
VisualPosOri1 = PyTree
VisualPosOri2 = PyTree


class Scene(ABC):
    _xyz: bool = True
    _xyz_root: bool = True
    visuals: list[Visual] = []

    """
    Example:
        >> renderer = Renderer()
        >> renderer.init(sys.geoms)
        >> for t in range(xs.shape()):
        >>   renderer.update(xs[t])
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

    def enable_xyz(self, enable_root: bool = True) -> None:
        self._xyz = True
        if enable_root:
            self._xyz_root = True

    def disable_xyz(self, disable_root: bool = True) -> None:
        self._xyz = False
        if disable_root:
            self._xyz_root = False

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
        raise NotImplementedError

    def _add_sphere(self, sphere: Sphere) -> Visual:
        raise NotImplementedError

    def _add_cylinder(self, cyl: Cylinder) -> Visual:
        raise NotImplementedError

    def _add_capsule(self, cap: Capsule) -> Visual:
        raise NotImplementedError

    def _add_xyz(self) -> Visual:
        raise NotImplementedError

    @abstractmethod
    def _remove_visual(self, visual: Visual) -> None:
        pass

    def _remove_all_visuals(self):
        for visual in self.visuals:
            self._remove_visual(visual)

    def init(self, geoms: list[Geometry]):
        self._remove_all_visuals()

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
            elif isinstance(geom, XYZ):
                visual = self._add_xyz()
            else:
                raise Exception(f"Unknown geom type: {type(geom)}")
            self.visuals.append(visual)

        if self._xyz:
            unique_link_indices = np.unique(np.array(geom_link_idx))
            for unique_link_idx in unique_link_indices:
                geom_link_idx.append(unique_link_idx)
                geom_transform.append(base.Transform.zero())
                self.visuals.append(self._add_xyz())
                # otherwise the .update function won't iterate
                # over all visuals since it uses a zip(...)
                self.geoms.append(None)

        if self._xyz_root:
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


class VispyScene(Scene):
    def __init__(
        self,
        show_cs=False,
        show_cs_root=True,
        size=(1280, 720),
        camera: scene.cameras.BaseCamera = scene.TurntableCamera(
            elevation=25, distance=4.0, azimuth=25
        ),
        **kwargs,
    ):
        """Scene which can be rendered.

        Args:
            geoms (list[list[Geometry]]): A list of list of geometries per link.
                len(geoms) == number of links in system
            show_cs (bool, optional): Show coordinate system of links.
                Defaults to True.
            show_cs_root (bool, optional): Show coordinate system of earth frame.
                Defaults to True.
            size (tuple, optional): Width and height of rendered image.
                Defaults to (1280, 720).
            camera (scene.cameras.BaseCamera, optional): The camera angle.
                Defaults to scene.TurntableCamera( elevation=30, distance=6 ).

        Example:
            >> scene = VispyScene()
            >> scene.init(sys.geoms)
            >> scene.update(state.x)
            >> image = scene.render()
        """
        self.canvas = scene.SceneCanvas(
            keys="interactive", size=size, show=True, **kwargs
        )
        self.view = self.canvas.central_widget.add_view()
        self._set_camera(camera)
        if show_cs:
            self.enable_xyz()
        else:
            self.disable_xyz(not show_cs_root)

    def _set_camera(self, camera: scene.cameras.BaseCamera) -> None:
        self.view.camera = camera

    def _get_camera(self) -> scene.cameras.BaseCamera:
        return self.view.camera

    def _render(self) -> jax.Array:
        return self.canvas.render(alpha=True)

    def _add_box(self, box: Box) -> Visual:
        return visuals.Box(
            box.dim_x,
            box.dim_z,
            box.dim_y,
            color=box.color,
            edge_color=box.edge_color,
            parent=self.view.scene,
        )

    def _add_sphere(self, sphere: Sphere) -> Visual:
        return visuals.Sphere(
            sphere.radius,
            color=sphere.color,
            edge_color=sphere.edge_color,
            parent=self.view.scene,
        )

    def _add_cylinder(self, cyl: Cylinder) -> Visual:
        return visuals.Cylinder(
            cyl.radius,
            cyl.length,
            color=cyl.color,
            edge_color=cyl.edge_color,
            parent=self.view.scene,
        )

    def _add_capsule(self, cap: Capsule) -> Visual:
        return visuals.Capsule(
            cap.radius,
            cap.length,
            color=cap.color,
            edge_color=cap.edge_color,
            parent=self.view.scene,
        )

    def _add_xyz(self) -> Visual:
        return scene.visuals.XYZAxis(parent=self.view.scene)

    def _remove_visual(self, visual: scene.visuals.VisualNode) -> None:
        visual.parent = None

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


def _animate_image(
    path: Union[str, Path],
    x: base.Transform,
    scene: Scene,
    *,
    fmt: Optional[str] = None,
    verbose: bool = True,
):
    scene.update(x)
    frame = scene.render()

    # remove alpha channel
    if fmt == "jpg" and frame.shape[-1] == 4:
        alpha = frame[..., 3]

        # assumes frame is u8
        if np.any(alpha != 255):
            raise Exception("jpg does not support alpha")

        frame = frame[..., :3]

    if verbose:
        print(f"Converting frames to {path} (this might take a while..)")

    imageio.imsave(path, frame, format=fmt)


def _animate_video(
    path: Union[str, Path],
    xs: list[base.Transform],
    scene: Scene,
    fps: int,
    N: int,
    step: int,
    *,
    show_pbar=True,
    fmt: Optional[str] = None,
    verbose: bool = True,
):
    frames = []
    for t in tqdm.tqdm(range(0, N, step), "Rendering frames..", disable=not show_pbar):
        scene.update(xs[t])
        frames.append(scene.render())

    if verbose:
        print(f"DONE. Converting frames to {path} (this might take a while..)")

    imageio.mimsave(path, frames, format=fmt, fps=fps)


def animate(
    path: Union[str, Path],
    sys: base.System,
    xs: base.Transform | Sequence[base.Transform],
    fps: int = 50,
    fmt: Optional[str] = None,
    verbose: bool = True,
    show_pbar: bool = True,
    **kwargs,
):
    """
    Make animation from system and trajectory of maximal coordinates. `xs` is either
    a single base.Transform object for images or a Sequence of base.Transform objects
    for a video format. The desired output format can be either inferred implicitely
    from the extension of `path` or set explicitely using the `fmt` parameter. Mismatch
    between the two is an error.
    """
    path = Path(path)
    file_fmt = _infer_extension_from_path(path)

    if file_fmt is not None and fmt is not None:
        assert (
            file_fmt == fmt.lower()
        ), f"""The chosen filename `{path.name}` and required fmt `{fmt}`
        are inconsistent."""
    elif file_fmt is None and fmt is not None:
        path = path.with_suffix("." + fmt)
    elif fmt is None and file_fmt is not None:
        fmt = file_fmt
    else:
        raise ValueError("neither fmt nor path extension given, can't infer format")

    scene = _init_vispy_scene(sys, **kwargs)

    n_links = sys.num_links()

    def data_check(x):
        assert (
            x.pos.ndim == x.rot.ndim == 2
        ), f"Expected shape = (n_links, 3/4). Got pos.shape{x.pos.shape}, "
        "rot.shape={x.rot.shape}"
        assert (
            x.pos.shape[0] == x.rot.shape[0] == n_links
        ), "Number of links does not match"

    if fmt in ["jpg", "png"]:
        # image fmts

        if isinstance(xs, base.Transform):
            x = xs
        else:
            x = xs[0]

        data_check(x)

        _animate_image(path, x, scene, fmt=fmt, verbose=verbose)

    elif fmt in ["mp4", "gif"]:
        # video fmts

        if isinstance(xs, base.Transform):
            xs = [xs]
        else:
            xs = list(xs)

        for x in xs:
            data_check(x)

        N = len(xs)
        _, step = _parse_timestep(sys.dt, fps, N)

        _animate_video(
            path, xs, scene, fps, N, step, show_pbar=show_pbar, fmt=fmt, verbose=verbose
        )
    else:
        raise ValueError(f"fmt {fmt} is not implement")


class Window:
    def __init__(
        self,
        sys: base.System,
        x: base.Transform,
        fps: int = 50,
        show_fps: bool = False,
        **kwargs,
    ):
        """Open an interactive Window that plays back the pre-computed trajectory.

        Args:
            scene (VispyScene): Scene used for rendering.
            x (base.Transform): Pre-computed trajectory.
            timestep (float): Timedelta between Transforms.
            fps (int, optional): Frame-rate. Defaults to 50.
        """
        self._x = x
        self._scene = _init_vispy_scene(sys, **kwargs)
        _data_checks(sys.num_links(), x.pos, x.rot)

        self.N = x.pos.shape[0]
        self.T, self.step = _parse_timestep(sys.dt, fps, self.N)
        self.timestep = sys.dt
        self.fps = fps
        self.show_fps = show_fps

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

        if self.show_fps:
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


def gui(
    sys: base.System,
    x: base.Transform,
    fps: int = 50,
    show_fps: bool = False,
    **kwargs,
):
    """Open an interactive Window that plays back the pre-computed trajectory.

    Args:
        scene (VispyScene): Scene used for rendering.
        x (base.Transform): Pre-computed trajectory.
        timestep (float): Timedelta between Transforms.
        fps (int, optional): Frame-rate. Defaults to 50.
    """
    if tree_utils.tree_ndim(x) == 2:
        x = x.batch()

    window = Window(sys, x, fps, show_fps, **kwargs)
    window.open()
    return window._scene.canvas


def probe(sys, **kwargs):
    state = base.State.create(sys)
    _, state = x_xy.forward_kinematics(sys, state)
    return gui(sys, state.x, **kwargs)


def _parse_timestep(timestep: float, fps: int, N: int):
    assert 1 / timestep > fps, "The `fps` is too high for the simulated timestep"
    fps_simu = int(1 / timestep)
    assert (fps_simu % fps) == 0, "The `fps` does not align with the timestep"
    T = N * timestep
    step = int(fps_simu / fps)
    return T, step


def _data_checks(n_links, data_pos, data_rot):
    assert (
        data_pos.ndim == data_rot.ndim == 3
    ), "Expected shape = (n_timesteps, n_links, 3/4)"
    assert (
        data_pos.shape[1] == data_rot.shape[1] == n_links
    ), "Number of links does not match"


def _infer_extension_from_path(path: Path) -> Optional[str]:
    ext = path.suffix
    # fmt starts after the . e.g. .mp4
    return ext[1:] if len(ext) > 0 else None


def _init_vispy_scene(sys: base.System, **kwargs) -> Scene:
    scene = VispyScene(**kwargs)
    scene.init(sys.geoms)
    return scene


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
