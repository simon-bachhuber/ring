from vispy.visuals import TubeVisual, CompoundVisual, SphereVisual, BaseVisual
from vispy.scene.visuals import create_visual_node, Sphere, Tube
from vispy.scene.node import Node
from vispy.visuals.transforms import STTransform

import jax.numpy as jnp


class CylinderVisual(TubeVisual):
    def __init__(self, radius, length, **kwargs):
        self.radius = radius
        self.length = length

        n_points = int(length * 100)

        tube_points = jnp.zeros((n_points, 3))
        tube_points = tube_points.at[:, 0].set(
            jnp.linspace(-length / 2, length / 2, n_points)
        )

        return super().__init__(
            tube_points,
            jnp.full((n_points,), radius),
            closed=True,
            **kwargs,
        )


Cylinder = create_visual_node(CylinderVisual)


class Capsule(Node):
    def __init__(self, radius, length, parent=None, **kwargs):
        super().__init__(parent=parent)

        color = kwargs.pop("color", None)

        n_points = int(length * 100)

        tube_points = jnp.zeros((n_points, 3))
        tube_points = tube_points.at[:, 0].set(
            jnp.linspace(-length / 2, length / 2, n_points)
        )

        sphere1 = Sphere(radius, parent=self, **kwargs, edge_color=color)
        sphere2 = Sphere(radius, parent=self, **kwargs, edge_color=color)
        tube = Tube(
            tube_points,
            jnp.full((n_points,), radius),
            parent=self,
            color=color,
            **kwargs,
        )

        self.radius = radius
        self.length = length

        self.sphere1 = sphere1
        self.sphere2 = sphere2
        self.tube = tube

        self.sphere1_shift = STTransform()
        self.sphere1_shift.move([length / 2, 0, 0])

        self.sphere2_shift = STTransform()
        self.sphere2_shift.move([-length / 2, 0, 0])

        self.sphere1.transform = self.sphere1_shift
        self.sphere2.transform = self.sphere2_shift

    """def _transform_changed(self, event=None):
        for v in self._subvisuals:
            v.transforms = self.transforms

        # patch sphere transforms by appending the respective shift
        # self.sphere1.transform = ChainTransform([self.sphere1.transform, self.sphere1_shift])
        # self.sphere2.transform = ChainTransform([self.sphere2.transform, self.sphere2_shift])

        # if self.transforms is not None:
        #     self.transforms.changed.disconnect(self._transform_changed)

        # if isinstance(self.sphere1.transform, ChainTransform):
        #     self.sphere1.transform.append(self.sphere1_shift)
        # elif isinstance(self.sphere1.transform, STTransform):
        #     self.sphere1.transform.move([self.length / 2, 0, 0])
        # else:
        #     self.sphere1.transform = ChainTransform(
        #         [self.sphere1.transform, self.sphere1_shift]
        #     )

        # if isinstance(self.sphere2.transform, ChainTransform):
        #     self.sphere2.transform.append(self.sphere2_shift)
        # elif isinstance(self.sphere2.transform, STTransform):
        #     self.sphere2.transform.move([-self.length / 2, 0, 0])
        # else:
        #     self.sphere2.transform = ChainTransform(
        #         [self.sphere2.transform, self.sphere2_shift]
        #     )

        # if self.transforms is not None:
        #     self.transforms.changed.connect(self._transform_changed)

        BaseVisual._transform_changed(self)"""


# Capsule = create_visual_node(CapsuleVisual)
