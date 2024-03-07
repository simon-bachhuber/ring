from abc import ABC
from abc import abstractmethod
from abc import abstractstaticmethod
from functools import partial
from typing import Optional, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from ring import algebra
from ring import base
from ring import maths
from tree_utils import PyTree
from tree_utils import tree_batch
from vispy import scene
from vispy.scene import MatrixTransform

from . import vispy_visuals

Camera = TypeVar("Camera")
Visual = TypeVar("Visual")
VisualPosOri1 = PyTree
VisualPosOri2 = PyTree


class Scene(ABC):
    _xyz: bool = True
    _xyz_root: bool = True
    _xyz_transform1: bool = True
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

    def enable_xyz_transform1(self):
        self._xyz_transform1 = True

    def disable_xyz_tranform1(self):
        self._xyz_transform1 = False

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

    def _add_box(self, box: base.Box) -> Visual:
        raise NotImplementedError

    def _add_sphere(self, sphere: base.Sphere) -> Visual:
        raise NotImplementedError

    def _add_cylinder(self, cyl: base.Cylinder) -> Visual:
        raise NotImplementedError

    def _add_capsule(self, cap: base.Capsule) -> Visual:
        raise NotImplementedError

    def _add_xyz(self) -> Visual:
        raise NotImplementedError

    @abstractmethod
    def _remove_visual(self, visual: Visual) -> None:
        pass

    def _remove_all_visuals(self):
        for visual in self.visuals:
            self._remove_visual(visual)

    def init(self, geoms: list[base.Geometry]):
        self._remove_all_visuals()

        self.geoms = [geom for geom in geoms]
        self._fresh_init = True

        geom_link_idx = []
        geom_transform = []
        self.visuals = []
        for geom in geoms:
            geom_link_idx.append(geom.link_idx)
            geom_transform.append(geom.transform)
            if isinstance(geom, base.Box):
                visual = self._add_box(geom)
            elif isinstance(geom, base.Sphere):
                visual = self._add_sphere(geom)
            elif isinstance(geom, base.Cylinder):
                visual = self._add_cylinder(geom)
            elif isinstance(geom, base.Capsule):
                visual = self._add_capsule(geom)
            elif isinstance(geom, base.XYZ):
                visual = self._add_xyz()
                if not self._xyz_transform1:
                    geom_transform.pop()
                    geom_transform.append(base.Transform.zero())
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
        self, visual: Visual, transform: VisualPosOri2, geom: None | base.Geometry
    ):
        pass

    def _update_visual(
        self, visual: Visual, transform: VisualPosOri2, geom: None | base.Geometry
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
        width: int = 320,
        height: int = 240,
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
            camera (scene.cameras.BaseCamera, optional): The camera angle.
                Defaults to scene.TurntableCamera( elevation=30, distance=6 ).

        Example:
            >> scene = VispyScene()
            >> scene.init(sys.geoms)
            >> scene.update(state.x)
            >> image = scene.render()
        """
        self.canvas = scene.SceneCanvas(
            keys="interactive", size=(width, height), show=True, **kwargs
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

    def _add_box(self, box: base.Box) -> Visual:
        return vispy_visuals.Box(
            box.dim_x,
            box.dim_z,
            box.dim_y,
            color=box.color,
            edge_color=box.edge_color,
            parent=self.view.scene,
        )

    def _add_sphere(self, sphere: base.Sphere) -> Visual:
        return vispy_visuals.Sphere(
            sphere.radius,
            color=sphere.color,
            edge_color=sphere.edge_color,
            parent=self.view.scene,
        )

    def _add_cylinder(self, cyl: base.Cylinder) -> Visual:
        return vispy_visuals.Cylinder(
            cyl.radius,
            cyl.length,
            color=cyl.color,
            edge_color=cyl.edge_color,
            parent=self.view.scene,
        )

    def _add_capsule(self, cap: base.Capsule) -> Visual:
        return vispy_visuals.Capsule(
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
        self,
        visual: scene.visuals.VisualNode,
        transform: np.ndarray,
        geom: base.Geometry,
    ):
        visual.transform = MatrixTransform(transform)

    def _update_visual(
        self,
        visual: scene.visuals.VisualNode,
        transform: np.ndarray,
        geom: base.Geometry,
    ):
        visual.transform.matrix = transform
