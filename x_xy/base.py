from dataclasses import field
from typing import Any, Sequence, Union

import jax
import jax.numpy as jnp
import tree_utils as tu
from flax import struct
from jax.tree_util import tree_map

from x_xy import maths

Scalar = jax.Array
Vector = jax.Array
Quaternion = jax.Array


class _Base:
    """Base functionality of all spatial datatypes.
    Copied and modified from https://github.com/google/brax/blob/main/brax/v2/base.py
    """

    def __add__(self, o: Any) -> Any:
        return tree_map(lambda x, y: x + y, self, o)

    def __sub__(self, o: Any) -> Any:
        return tree_map(lambda x, y: x - y, self, o)

    def __mul__(self, o: Any) -> Any:
        return tree_map(lambda x: x * o, self)

    def __neg__(self) -> Any:
        return tree_map(lambda x: -x, self)

    def __truediv__(self, o: Any) -> Any:
        return tree_map(lambda x: x / o, self)

    def __getitem__(self, i: int) -> Any:
        return self.take(i)

    def reshape(self, shape: Sequence[int]) -> Any:
        return tree_map(lambda x: x.reshape(shape), self)

    def slice(self, beg: int, end: int) -> Any:
        return tree_map(lambda x: x[beg:end], self)

    def take(self, i, axis=0) -> Any:
        return tree_map(lambda x: jnp.take(x, i, axis=axis), self)

    def hstack(self, *others: Any) -> Any:
        return tree_map(lambda *x: jnp.hstack(x), self, *others)

    def vstack(self, *others: Any) -> Any:
        return tree_map(lambda *x: jnp.vstack(x), self, *others)

    def concatenate(self, *others: Any, axis: int = 0) -> Any:
        return tree_map(lambda *x: jnp.concatenate(x, axis=axis), self, *others)

    def batch(self, *others, along_existing_first_axis: bool = False) -> Any:
        return tu.tree_batch((self,) + others, along_existing_first_axis, "jax")

    def index_set(self, idx: Union[jnp.ndarray, Sequence[jnp.ndarray]], o: Any) -> Any:
        return tree_map(lambda x, y: x.at[idx].set(y), self, o)

    def index_sum(self, idx: Union[jnp.ndarray, Sequence[jnp.ndarray]], o: Any) -> Any:
        return tree_map(lambda x, y: x.at[idx].add(y), self, o)

    @property
    def T(self):
        return tree_map(lambda x: x.T, self)

    def flatten(self, num_batch_dims: int = 0) -> jax.Array:
        return tu.batch_concat(self, num_batch_dims)

    def squeeze(self):
        return tree_map(lambda x: jnp.squeeze(x), self)

    def squeeze_1d(self):
        return tree_map(lambda x: jnp.atleast_1d(jnp.squeeze(x)), self)


@struct.dataclass
class Transform(_Base):
    """Represents the Transformation from Plücker A to Plücker B,
    where B is located relative to A at `pos` in frame A and `rot` is the
    relative quaternion from A to B."""

    pos: Vector
    rot: Quaternion

    @classmethod
    def create(cls, pos=None, rot=None):
        assert not (pos is None and rot is None), "One must be given."
        if pos is None:
            pos = jnp.zeros((3,))
        if rot is None:
            rot = jnp.array([1.0, 0, 0, 0])
        return Transform(pos, rot)

    @classmethod
    def zero(cls, shape=()) -> "Transform":
        """Returns a zero transform with a batch shape."""
        pos = jnp.zeros(shape + (3,))
        rot = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), shape + (1,))
        return Transform(pos, rot)

    def as_matrix(self) -> jax.Array:
        E = maths.quat_to_3x3(self.rot)
        return maths.spatial.quadrants(aa=E, bb=E) @ maths.spatial.xlt(self.pos)


@struct.dataclass
class Motion(_Base):
    "Coordinate vector that represents a spatial motion vector in Plücker Coordinates."
    ang: Vector
    vel: Vector

    @classmethod
    def create(cls, ang=None, vel=None):
        assert not (ang is None and vel is None), "One must be given."
        if ang is None:
            ang = jnp.zeros((3,))
        if vel is None:
            vel = jnp.zeros((3,))
        return Motion(ang, vel)

    @classmethod
    def zero(cls, shape=()) -> "Motion":
        ang = jnp.zeros(shape + (3,))
        vel = jnp.zeros(shape + (3,))
        return Motion(ang, vel)

    def as_matrix(self):
        return self.flatten()


@struct.dataclass
class Force(_Base):
    "Coordinate vector that represents a spatial force vector in Plücker Coordinates."
    ang: Vector
    vel: Vector

    @classmethod
    def create(cls, ang=None, vel=None):
        assert not (ang is None and vel is None), "One must be given."
        if ang is None:
            ang = jnp.zeros((3,))
        if vel is None:
            vel = jnp.zeros((3,))
        return Force(ang, vel)

    @classmethod
    def zero(cls, shape=()) -> "Force":
        ang = jnp.zeros(shape + (3,))
        vel = jnp.zeros(shape + (3,))
        return Force(ang, vel)

    def as_matrix(self):
        return self.flatten()


@struct.dataclass
class Inertia(_Base):
    """Spatial Inertia Matrix in Plücker Coordinates.
    Note that `h` is *not* the center of mass."""

    it_3x3: jax.Array
    h: Vector
    mass: Vector

    @classmethod
    def create(cls, mass: Vector, CoM: Vector, it_3x3: jnp.ndarray):
        """Construct spatial inertia of an object with mass `mass` located
        at the center of mass `CoM` and an inertia matrix `it_3x3` around that
        center of mass.
        """
        it_3x3 = maths.spatial.mcI(mass, CoM, it_3x3)[:3, :3]
        h = mass * CoM
        return Inertia(it_3x3, h, mass)

    @classmethod
    def zero(cls, shape=()) -> "Inertia":
        it_shape_3x3 = jnp.zeros(shape + (3, 3))
        h = jnp.zeros(shape + (3,))
        mass = jnp.zeros(shape + (1,))
        return Inertia(it_shape_3x3, h, mass)

    def as_matrix(self):
        hcross = maths.spatial.cross(self.h)
        return maths.spatial.quadrants(
            self.it_3x3, hcross, -hcross, self.mass * jnp.eye(3)
        )


@struct.dataclass
class Geometry(_Base):
    mass: jax.Array
    CoM: jax.Array


@struct.dataclass
class Sphere(Geometry):
    radius: jax.Array
    vispy_kwargs: dict = field(default_factory=lambda: {})

    def get_it_3x3(self) -> jax.Array:
        it_3x3 = 2 / 5 * self.mass * self.radius**2 * jnp.eye(3)
        return it_3x3


@struct.dataclass
class Box(Geometry):
    dim_x: jax.Array
    dim_y: jax.Array
    dim_z: jax.Array
    vispy_kwargs: dict = field(default_factory=lambda: {})

    def get_it_3x3(self) -> jax.Array:
        it_3x3 = (
            1
            / 12
            * self.mass
            * jnp.array(
                [
                    [self.dim_y**2 + self.dim_z**2, 0, 0],
                    [0, self.dim_x**2 + self.dim_z**2, 0],
                    [0, 0, self.dim_x**2 + self.dim_y**2],
                ]
            )
        )
        return it_3x3


@struct.dataclass
class Cylinder(Geometry):
    """Length is along x-axis."""

    radius: jax.Array
    length: jax.Array
    vispy_kwargs: dict = field(default_factory=lambda: {})

    def get_it_3x3(self) -> jax.Array:
        radius_dir = 3 * self.radius**2 + self.length**2
        it_3x3 = (
            1
            / 12
            * self.mass
            * jnp.array(
                [
                    [6 * self.radius**2, 0, 0],
                    [0, radius_dir, 0],
                    [0, 0, radius_dir],
                ]
            )
        )
        return it_3x3


def inertia_from_geometries(geometries: list[Geometry]) -> Inertia:
    inertia = Inertia.zero()
    for geom in geometries:
        inertia += Inertia.create(geom.mass, geom.CoM, geom.get_it_3x3())
    return inertia


N_JOINT_PARAMS: int = 1


@struct.dataclass
class Link(_Base):
    transform1: Transform

    # these parameters can be used to model joints that have parameters
    # they are directly feed into the `jcalc` routine
    # this array *must* be of shape (N_JOINT_PARAMS,)
    joint_params: jax.Array = jnp.zeros((N_JOINT_PARAMS,))

    # internal useage
    inertia: Inertia = Inertia.zero()
    transform2: Transform = Transform.zero()
    transform: Transform = Transform.zero()


@struct.dataclass
class System(_Base):
    link_parents: jax.Array
    links: Link
    link_joint_types: list[str] = struct.field(False)

    # simulation timestep size
    dt: float = struct.field(False)

    # whether or not to re-calculate the inertia
    # matrix at every simulation timestep because
    # the geometries may have changed
    dynamic_geometries: bool = struct.field(False)

    # root / base acceleration offset
    gravity: jax.Array = jnp.array([0, 0, 9.81])

    @property
    def parent(self) -> jax.Array:
        return self.link_parents

    @property
    def N(self):
        return len(self.link_parents)


@struct.dataclass
class State(_Base):
    q: dict[int, jax.Array]
    alpha: dict[int, jax.Array]
    x: Transform
    xd: Motion
