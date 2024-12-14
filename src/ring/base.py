from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

from flax import struct
import jax
from jax.core import Tracer
import jax.numpy as jnp
from jax.tree_util import tree_map
import numpy as np
import tree
import tree_utils as tu

import ring
from ring import maths
from ring import spatial

Scalar = jax.Array
Vector = jax.Array
Quaternion = jax.Array


Color = Optional[str | tuple[float, float, float] | tuple[float, float, float, float]]


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

    def batch_dim(self) -> int:
        return tu.tree_shape(self)

    def transpose(self, axes: Sequence[int]) -> Any:
        return tree_map(lambda x: jnp.transpose(x, axes), self)

    def __iter__(self):
        raise NotImplementedError

    def repeat(self, repeats, axis=0):
        return tree_map(lambda x: jnp.repeat(x, repeats, axis), self)

    def ndim(self):
        return tu.tree_ndim(self)

    def shape(self, axis: int = 0) -> int:
        Bs = tree_map(lambda arr: arr.shape[axis], self)
        Bs = set(jax.tree_util.tree_flatten(Bs)[0])
        assert len(Bs) == 1
        return list(Bs)[0]

    def __len__(self) -> int:
        return self.shape(axis=0)


@struct.dataclass
class Transform(_Base):
    """Represents the Transformation from Plücker A to Plücker B,
    where B is located relative to A at `pos` in frame A and `rot` is the
    relative quaternion from A to B.
    Create using `Transform.create(pos=..., rot=...)
    """

    pos: Vector
    rot: Quaternion

    @classmethod
    def create(cls, pos=None, rot=None):
        assert not (pos is None and rot is None), "One must be given."
        shape_rot = rot.shape[:-1] if rot is not None else ()
        shape_pos = pos.shape[:-1] if pos is not None else ()

        if pos is None:
            pos = jnp.zeros(shape_rot + (3,))
        if rot is None:
            rot = jnp.array([1.0, 0, 0, 0])
            rot = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), shape_pos + (1,))

        assert pos.shape[:-1] == rot.shape[:-1]

        return Transform(pos, rot)

    @classmethod
    def zero(cls, shape=()) -> "Transform":
        """Returns a zero transform with a batch shape."""
        pos = jnp.zeros(shape + (3,))
        rot = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), shape + (1,))
        return Transform(pos, rot)

    def as_matrix(self) -> jax.Array:
        E = maths.quat_to_3x3(self.rot)
        return spatial.quadrants(aa=E, bb=E) @ spatial.xlt(self.pos)


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
    def create(cls, mass: Vector, transform: Transform, it_3x3: jnp.ndarray):
        """Construct spatial inertia of an object with mass `mass` located and aligned
        with a coordinate system that is given by `transform` where `transform` is from
        parent to local geometry coordinates.
        """
        it_3x3 = maths.rotate_matrix(it_3x3, maths.quat_inv(transform.rot))
        it_3x3 = spatial.mcI(mass, transform.pos, it_3x3)[:3, :3]
        h = mass * transform.pos
        return cls(it_3x3, h, mass)

    @classmethod
    def zero(cls, shape=()) -> "Inertia":
        it_shape_3x3 = jnp.zeros(shape + (3, 3))
        h = jnp.zeros(shape + (3,))
        mass = jnp.zeros(shape + (1,))
        return cls(it_shape_3x3, h, mass)

    def as_matrix(self):
        hcross = spatial.cross(self.h)
        return spatial.quadrants(self.it_3x3, hcross, -hcross, self.mass * jnp.eye(3))


@struct.dataclass
class Geometry(_Base):
    mass: jax.Array
    transform: Transform
    link_idx: int = struct.field(pytree_node=False)

    color: Color = struct.field(pytree_node=False)
    edge_color: Color = struct.field(pytree_node=False)


@struct.dataclass
class XYZ(Geometry):
    # TODO: possibly subclass this of _Base? does this need a mass, transform, and
    # link_idx? maybe just transform?
    size: float

    @classmethod
    def create(cls, link_idx: int, size: float):
        return cls(0.0, Transform.zero(), link_idx, None, None, size)

    def get_it_3x3(self) -> jax.Array:
        return jnp.zeros((3, 3))


@struct.dataclass
class Sphere(Geometry):
    radius: float

    def get_it_3x3(self) -> jax.Array:
        it_3x3 = 2 / 5 * self.mass * self.radius**2 * jnp.eye(3)
        return it_3x3


@struct.dataclass
class Box(Geometry):
    dim_x: float
    dim_y: float
    dim_z: float

    def get_it_3x3(self) -> jax.Array:
        it_3x3 = (
            1
            / 12
            * self.mass
            * jnp.diag(
                jnp.array(
                    [
                        self.dim_y**2 + self.dim_z**2,
                        self.dim_x**2 + self.dim_z**2,
                        self.dim_x**2 + self.dim_y**2,
                    ]
                )
            )
        )
        return it_3x3


@struct.dataclass
class Cylinder(Geometry):
    """Length is along x-axis."""

    radius: float
    length: float

    def get_it_3x3(self) -> jax.Array:
        radius_dir = 3 * self.radius**2 + self.length**2
        it_3x3 = (
            1
            / 12
            * self.mass
            * jnp.diag(jnp.array([6 * self.radius**2, radius_dir, radius_dir]))
        )
        return it_3x3


@struct.dataclass
class Capsule(Geometry):
    """Length is along x-axis."""

    radius: float
    length: float

    def get_it_3x3(self) -> jax.Array:
        """https://github.com/thomasmarsh/ODE/blob/master/ode/src/mass.cpp#L141"""
        r = self.radius
        d = self.length

        v_cyl = jnp.pi * r**2 * d
        v_cap = 4 / 3 * jnp.pi * r**3

        v_tot = v_cyl + v_cap

        m_cyl = self.mass * v_cyl / v_tot
        m_cap = self.mass * v_cap / v_tot

        I_a = m_cyl * (0.25 * r**2 + 1 / 12 * d**2) + m_cap * (
            0.4 * r**2 + 0.375 * r * d + 0.25 * d**2
        )
        I_b = (0.5 * m_cyl + 0.4 * m_cap) * r**2

        return jnp.diag(jnp.array([I_b, I_a, I_a]))


_DEFAULT_JOINT_PARAMS_DICT: dict[str, tu.PyTree] = {"default": jnp.array([])}


@struct.dataclass
class Link(_Base):
    transform1: Transform

    # only used by `setup_fn_randomize_positions`
    pos_min: jax.Array = struct.field(default_factory=lambda: jnp.zeros((3,)))
    pos_max: jax.Array = struct.field(default_factory=lambda: jnp.zeros((3,)))

    # these parameters can be used to model joints that have parameters
    # they are directly feed into the `jcalc` routines
    joint_params: dict[str, tu.PyTree] = struct.field(
        default_factory=lambda: _DEFAULT_JOINT_PARAMS_DICT
    )

    # internal useage
    # gets populated by `parse_system`
    inertia: Inertia = Inertia.zero()
    # gets populated by `forward_kinematics`
    transform2: Transform = Transform.zero()
    transform: Transform = Transform.zero()


@struct.dataclass
class MaxCoordOMC(_Base):
    coordinate_system_name: str = struct.field(False)
    pos_marker_number: int = struct.field(False)
    pos_marker_constant_offset: jax.Array


Q_WIDTHS = {
    "free": 7,
    "free_2d": 3,
    "frozen": 0,
    "spherical": 4,
    "p3d": 3,
    # center of rotation, a `free` joint and then a `p3d` joint with custom
    # parameter fields in `RMCG_Config`
    "cor": 10,
    "px": 1,
    "py": 1,
    "pz": 1,
    "rx": 1,
    "ry": 1,
    "rz": 1,
    "saddle": 2,
}
QD_WIDTHS = {
    "free": 6,
    "free_2d": 3,
    "frozen": 0,
    "spherical": 3,
    "p3d": 3,
    "cor": 9,
    "px": 1,
    "py": 1,
    "pz": 1,
    "rx": 1,
    "ry": 1,
    "rz": 1,
    "saddle": 2,
}


@struct.dataclass
class System(_Base):
    "System object. Create using `System.create(path_xml)`"
    link_parents: list[int] = struct.field(False)
    links: Link
    link_types: list[str] = struct.field(False)
    link_damping: jax.Array
    link_armature: jax.Array
    link_spring_stiffness: jax.Array
    link_spring_zeropoint: jax.Array
    # simulation timestep size
    dt: float = struct.field(False)
    # geometries in the system
    geoms: list[Geometry]
    # root / base acceleration offset
    gravity: jax.Array = struct.field(default_factory=lambda: jnp.array([0, 0, -9.81]))

    integration_method: str = struct.field(
        False, default_factory=lambda: "semi_implicit_euler"
    )
    mass_mat_iters: int = struct.field(False, default_factory=lambda: 0)

    link_names: list[str] = struct.field(False, default_factory=lambda: [])

    model_name: Optional[str] = struct.field(False, default_factory=lambda: None)

    omc: list[MaxCoordOMC | None] = struct.field(True, default_factory=lambda: [])

    def num_links(self) -> int:
        return len(self.link_parents)

    def q_size(self) -> int:
        return sum([Q_WIDTHS[typ] for typ in self.link_types])

    def qd_size(self) -> int:
        return sum([QD_WIDTHS[typ] for typ in self.link_types])

    def name_to_idx(self, name: str) -> int:
        return self.link_names.index(name)

    def idx_to_name(self, idx: int, allow_world: bool = False) -> str:
        if allow_world and idx == -1:
            return "world"
        assert idx >= 0, "Worldbody index has no name."
        return self.link_names[idx]

    def idx_map(self, type: str) -> dict:
        "type: is either `l` or `q` or `d`"
        dict_int_slices = {}

        def f(_, idx_map, name: str, link_idx: int):
            dict_int_slices[name] = idx_map[type](link_idx)

        self.scan(f, "ll", self.link_names, list(range(self.num_links())))

        return dict_int_slices

    def parent_name(self, name: str) -> str:
        return self.idx_to_name(self.link_parents[self.name_to_idx(name)])

    def add_prefix(self, prefix: str = "") -> "System":
        return self.replace(link_names=[prefix + name for name in self.link_names])

    def change_model_name(
        self,
        new_name: Optional[str] = None,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ) -> "System":
        if prefix is None:
            prefix = ""
        if suffix is None:
            suffix = ""
        if new_name is None:
            new_name = self.model_name
        name = prefix + new_name + suffix
        return self.replace(model_name=name)

    def change_link_name(self, old_name: str, new_name: str) -> "System":
        old_idx = self.name_to_idx(old_name)
        new_link_names = self.link_names.copy()
        new_link_names[old_idx] = new_name
        return self.replace(link_names=new_link_names)

    def add_prefix_suffix(
        self, prefix: Optional[str] = None, suffix: Optional[str] = None
    ) -> "System":
        if prefix is None:
            prefix = ""
        if suffix is None:
            suffix = ""
        new_link_names = [prefix + name + suffix for name in self.link_names]
        return self.replace(link_names=new_link_names)

    def _replace_free_with_cor(self) -> "System":
        # check that
        # - all free joints connect to -1
        # - all joints connecting to -1 are free joints
        for i, p in enumerate(self.link_parents):
            link_type = self.link_types[i]
            if (p == -1 and link_type != "free") or (link_type == "free" and p != -1):
                raise InvalidSystemError(
                    f"link={self.idx_to_name(i)}, parent="
                    f"{self.idx_to_name(p, allow_world=True)},"
                    f" joint={link_type}. Hint: Try setting `config.cor` to false."
                )

        def logic_replace_free_with_cor(name, olt, ola, old, ols, olz):
            # by default new is equal to old
            nlt, nla, nld, nls, nlz = olt, ola, old, ols, olz

            # old link type == free
            if olt == "free":
                # cor joint is (free, p3d) stacked
                nlt = "cor"
                # entries of old armature are 3*ang (spherical), 3*pos (p3d)
                nla = jnp.concatenate((ola, ola[3:]))
                nld = jnp.concatenate((old, old[3:]))
                nls = jnp.concatenate((ols, ols[3:]))
                nlz = jnp.concatenate((olz, olz[4:]))

            return nlt, nla, nld, nls, nlz

        return _update_sys_if_replace_joint_type(self, logic_replace_free_with_cor)

    def freeze(self, name: str | list[str]):
        if isinstance(name, list):
            sys = self
            for n in name:
                sys = sys.freeze(n)
            return sys

        def logic_freeze(link_name, olt, ola, old, ols, olz):
            nlt, nla, nld, nls, nlz = olt, ola, old, ols, olz

            if link_name == name:
                nlt = "frozen"
                nla = nld = nls = nlz = jnp.array([])

            return nlt, nla, nld, nls, nlz

        return _update_sys_if_replace_joint_type(self, logic_freeze)

    def unfreeze(self, name: str, new_joint_type: str):
        assert self.link_types[self.name_to_idx(name)] == "frozen"
        assert new_joint_type != "frozen"

        return self.change_joint_type(name, new_joint_type)

    def change_joint_type(
        self,
        name: str,
        new_joint_type: str,
        new_arma: Optional[jax.Array] = None,
        new_damp: Optional[jax.Array] = None,
        new_stif: Optional[jax.Array] = None,
        new_zero: Optional[jax.Array] = None,
        seed: int = 1,
        warn: bool = True,
    ):
        "By default damping, stiffness are set to zero."
        from ring.algorithms import get_joint_model

        q_size, qd_size = Q_WIDTHS[new_joint_type], QD_WIDTHS[new_joint_type]

        def logic_unfreeze_to_spherical(link_name, olt, ola, old, ols, olz):
            nlt, nla, nld, nls, nlz = olt, ola, old, ols, olz

            if link_name == name:
                nlt = new_joint_type
                q_zeros = jnp.zeros((q_size))
                qd_zeros = jnp.zeros((qd_size,))

                nla = qd_zeros if new_arma is None else new_arma
                nld = qd_zeros if new_damp is None else new_damp
                nls = qd_zeros if new_stif is None else new_stif
                nlz = q_zeros if new_zero is None else new_zero

                # unit quaternion
                if new_joint_type in ["spherical", "free", "cor"] and new_zero is None:
                    nlz = nlz.at[0].set(1.0)

            return nlt, nla, nld, nls, nlz

        sys = _update_sys_if_replace_joint_type(self, logic_unfreeze_to_spherical)

        jm = get_joint_model(new_joint_type)
        if jm.init_joint_params is not None:
            sys = sys.from_str(sys.to_str(warn=warn), seed=seed)

        return sys

    @staticmethod
    def joint_type_simplification(typ: str) -> str:
        if typ[:4] == "free":
            if typ == "free_2d":
                return "free_2d"
            else:
                return "free"
        elif typ[:3] == "cor":
            return "cor"
        elif typ[:9] == "spherical":
            return "spherical"
        else:
            return typ

    @staticmethod
    def joint_type_is_free_or_cor(typ: str) -> bool:
        return System.joint_type_simplification(typ) in ["free", "cor"]

    @staticmethod
    def joint_type_is_spherical(typ: str) -> bool:
        return System.joint_type_simplification(typ) == "spherical"

    @staticmethod
    def joint_type_is_free_or_cor_or_spherical(typ: str) -> bool:
        return System.joint_type_is_free_or_cor(typ) or System.joint_type_is_spherical(
            typ
        )

    def findall_imus(self, names: bool = True) -> list[str] | list[int]:
        bodies = [name for name in self.link_names if name[:3] == "imu"]
        return bodies if names else [self.name_to_idx(n) for n in bodies]

    def findall_segments(self, names: bool = True) -> list[str] | list[int]:
        imus = self.findall_imus(names=True)
        bodies = [name for name in self.link_names if name not in imus]
        return bodies if names else [self.name_to_idx(n) for n in bodies]

    def _bodies_indices_to_bodies_name(self, bodies: list[int]) -> list[str]:
        return [self.idx_to_name(i) for i in bodies]

    def findall_bodies_to_world(self, names: bool = False) -> list[int] | list[str]:
        bodies = [i for i, p in enumerate(self.link_parents) if p == -1]
        return self._bodies_indices_to_bodies_name(bodies) if names else bodies

    def find_body_to_world(self, name: bool = False) -> int | str:
        bodies = self.findall_bodies_to_world(names=name)
        assert len(bodies) == 1
        return bodies[0]

    def findall_bodies_with_jointtype(
        self, typ: str, names: bool = False
    ) -> list[int] | list[str]:
        bodies = [i for i, _typ in enumerate(self.link_types) if _typ == typ]
        return self._bodies_indices_to_bodies_name(bodies) if names else bodies

    def children(self, name: str, names: bool = False) -> list[int] | list[str]:
        "List all direct children of body, does not include body itself"
        p = self.name_to_idx(name)
        bodies = [i for i in range(self.num_links()) if self.link_parents[i] == p]
        return bodies if (not names) else [self.idx_to_name(i) for i in bodies]

    def findall_bodies_subsystem(
        self, name: str, names: bool = False
    ) -> list[int] | list[str]:
        "List all children and children's children; does not include body itself"
        children = self.children(name, names=True)
        grandchildren = [self.findall_bodies_subsystem(n, names=True) for n in children]
        bodies = tree.flatten([children, grandchildren])
        return bodies if names else [self.name_to_idx(n) for n in bodies]

    def scan(self, f: Callable, in_types: str, *args, reverse: bool = False):
        """Scan `f` along each link in system whilst carrying along state.

        Args:
            f (Callable[..., Y]): f(y: Y, *args) -> y
            in_types: string specifying the type of each input arg:
                'l' is an input to be split according to link ranges
                'q' is an input to be split according to q ranges
                'd' is an input to be split according to qd ranges
            args: Arguments passed to `f`, and split to match the link.
            reverse (bool, optional): If `true` from leaves to root. Defaults to False.

        Returns:
            ys: Stacked output y of f.
        """
        return _scan_sys(self, f, in_types, *args, reverse=reverse)

    def parse(self) -> "System":
        """Initial setup of system. System object does not work unless it is parsed.
        Currently it does:
        - some consistency checks
        - populate the spatial inertia tensors
        - check that all names are unique
        - check that names are strings
        - check that all pos_min <= pos_max (unless traced)
        - order geoms in ascending order based on their parent link idx
        - check that all links have the correct size of
            - damping
            - armature
            - stiffness
            - zeropoint
        - check that n_links == len(sys.omc)
        """
        return _parse_system(self)

    def render(
        self,
        xs: Optional[Transform | list[Transform]] = None,
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
        return ring.rendering.render(
            self, xs, camera, show_pbar, backend, render_every_nth, **scene_kwargs
        )

    def render_prediction(
        self,
        xs: Transform | list[Transform],
        yhat: dict | jax.Array | np.ndarray,
        # by default we don't predict the global rotation
        transparent_segment_to_root: bool = True,
        **kwargs,
    ):
        """`xs` matches `sys`. `yhat` matches `sys_noimu`. `yhat` are child-to-parent.
        Note that the body in yhat that connects to -1, is parent-to-child!
        """
        return ring.rendering.render_prediction(
            self, xs, yhat, transparent_segment_to_root, **kwargs
        )

    def delete_system(self, link_name: str | list[str], strict: bool = True):
        "Cut subsystem starting at `link_name` (inclusive) from tree."
        return ring.sys_composer.delete_subsystem(self, link_name, strict)

    def make_sys_noimu(self, imu_link_names: Optional[list[str]] = None):
        "Returns, e.g., imu_attachment = {'imu1': 'seg1', 'imu2': 'seg3'}"
        return ring.sys_composer.make_sys_noimu(self, imu_link_names)

    def inject_system(self, other_system: "System", at_body: Optional[str] = None):
        """Combine two systems into one.

        Args:
            sys (base.System): Large system.
            sub_sys (base.System): Small system that will be included into the
                large system `sys`.
            at_body (Optional[str], optional): Into which body of the large system
                small system will be included. Defaults to `worldbody`.

        Returns:
            base.System: _description_
        """
        return ring.sys_composer.inject_system(self, other_system, at_body)

    def morph_system(
        self,
        new_parents: Optional[list[int | str]] = None,
        new_anchor: Optional[int | str] = None,
    ):
        """Re-orders the graph underlying the system. Returns a new system.

        Args:
            sys (base.System): System to be modified.
            new_parents (list[int]): Let the i-th entry have value j. Then, after
                morphing the system the system will be such that the link corresponding
                to the i-th link in the old system will have as parent the link
                corresponding to the j-th link in the old system.

        Returns:
            base.System: Modified system.
        """
        return ring.sys_composer.morph_system(self, new_parents, new_anchor)

    @staticmethod
    def from_xml(path: str, seed: int = 1):
        return ring.io.load_sys_from_xml(path, seed)

    @staticmethod
    def from_str(xml: str, seed: int = 1):
        return ring.io.load_sys_from_str(xml, seed)

    def to_str(self, warn: bool = True) -> str:
        return ring.io.save_sys_to_str(self, warn=warn)

    def to_xml(self, path: str) -> None:
        ring.io.save_sys_to_xml(self, path)

    @classmethod
    def create(cls, path_or_str: str, seed: int = 1) -> "System":
        path = Path(path_or_str).with_suffix(".xml")

        exists = False
        try:
            exists = path.exists()
        except OSError:
            # file length too length
            pass

        if exists:
            return cls.from_xml(path, seed=seed)
        else:
            return cls.from_str(path_or_str)

    def coordinate_vector_to_q(
        self,
        q: jax.Array,
        custom_joints: dict[str, Callable] = {},
    ) -> jax.Array:
        """Map a coordinate vector `q` to the minimal coordinates vector of the sys"""
        # Does, e.g.
        # - normalize quaternions
        # - hinge joints in [-pi, pi]
        q_preproc = []

        def preprocess(_, __, link_type, q):
            to_q = ring.algorithms.jcalc.get_joint_model(
                link_type
            ).coordinate_vector_to_q
            # function in custom_joints has priority over JointModel
            if link_type in custom_joints:
                to_q = custom_joints[link_type]
            if to_q is None:
                raise NotImplementedError(
                    f"Please specify the custom joint `{link_type}`"
                    " either using the `custom_joints` arguments or using the"
                    " JointModel.coordinate_vector_to_q field."
                )
            new_q = to_q(q)
            q_preproc.append(new_q)

        self.scan(preprocess, "lq", self.link_types, q)
        return jnp.concatenate(q_preproc)


def _update_sys_if_replace_joint_type(sys: System, logic) -> System:
    lt, la, ld, ls, lz = [], [], [], [], []

    def f(_, __, name, olt, ola, old, ols, olz):
        nlt, nla, nld, nls, nlz = logic(name, olt, ola, old, ols, olz)

        lt.append(nlt)
        la.append(nla)
        ld.append(nld)
        ls.append(nls)
        lz.append(nlz)

    sys.scan(
        f,
        "lldddq",
        sys.link_names,
        sys.link_types,
        sys.link_armature,
        sys.link_damping,
        sys.link_spring_stiffness,
        sys.link_spring_zeropoint,
    )

    # lt is supposed to be a list of strings; no concat required
    la, ld, ls, lz = map(jnp.concatenate, (la, ld, ls, lz))

    sys = sys.replace(
        link_types=lt,
        link_armature=la,
        link_damping=ld,
        link_spring_stiffness=ls,
        link_spring_zeropoint=lz,
    )

    # parse system such that it checks if all joint types have the
    # correct dimensionality of damping / stiffness / zeropoint / armature
    return sys.parse()


class InvalidSystemError(Exception):
    pass


def _parse_system(sys: System) -> System:
    assert len(sys.link_parents) == len(sys.link_types) == sys.links.batch_dim()
    assert len(sys.omc) == sys.num_links()

    for i, name in enumerate(sys.link_names):
        assert sys.link_names.count(name) == 1, f"Duplicated name=`{name}` in system"
        assert isinstance(name, str)

    pos_min, pos_max = sys.links.pos_min, sys.links.pos_max

    try:
        from jax.errors import TracerBoolConversionError

        try:
            assert jnp.all(pos_max >= pos_min), f"min={pos_min}, max={pos_max}"
        except TracerBoolConversionError:
            pass
    # on older versions of jax this import is not possible
    except ImportError:
        pass

    for geom in sys.geoms:
        assert geom.link_idx in list(range(sys.num_links())) + [-1]

    inertia = _parse_system_calculate_inertia(sys)
    sys = sys.replace(links=sys.links.replace(inertia=inertia))

    # sort geoms in ascending order
    geoms = sys.geoms.copy()
    geoms.sort(key=lambda geom: geom.link_idx)
    sys = sys.replace(geoms=geoms)

    # round dt
    # sys = sys.replace(dt=round(sys.dt, 8))

    # check sizes of damping / arma / stiff / zeropoint
    def check_dasz_unitq(_, __, name, typ, d, a, s, z):
        q_size, qd_size = Q_WIDTHS[typ], QD_WIDTHS[typ]

        error_msg = (
            f"wrong size for link `{name}` of typ `{typ}` in model {sys.model_name}"
        )

        assert d.size == a.size == s.size == qd_size, error_msg
        assert z.size == q_size, error_msg

        if System.joint_type_is_free_or_cor_or_spherical(typ) and not isinstance(
            z, Tracer
        ):
            assert jnp.allclose(
                jnp.linalg.norm(z[:4]), 1.0
            ), f"not unit quat for link `{name}` of typ `{typ}` in model"
            f" {sys.model_name}"

    sys.scan(
        check_dasz_unitq,
        "lldddq",
        sys.link_names,
        sys.link_types,
        sys.link_damping,
        sys.link_armature,
        sys.link_spring_stiffness,
        sys.link_spring_zeropoint,
    )

    return sys


def _inertia_from_geometries(geometries: list[Geometry]) -> Inertia:
    inertia = Inertia.zero()
    for geom in geometries:
        inertia += Inertia.create(geom.mass, geom.transform, geom.get_it_3x3())
    return inertia


def _parse_system_calculate_inertia(sys: System):
    def compute_inertia_per_link(_, __, link_idx: int):
        geoms_link = []
        for geom in sys.geoms:
            if geom.link_idx == link_idx:
                geoms_link.append(geom)

        it = _inertia_from_geometries(geoms_link)
        return it

    return sys.scan(compute_inertia_per_link, "l", list(range(sys.num_links())))


def _scan_sys(sys: System, f: Callable, in_types: str, *args, reverse: bool = False):
    assert len(args) == len(in_types)
    for in_type, arg in zip(in_types, args):

        if in_type == "l":
            required_length = sys.num_links()
        elif in_type == "q":
            required_length = sys.q_size()
        elif in_type == "d":
            required_length = sys.qd_size()
        else:
            raise Exception("`in_types` must be one of `l` or `q` or `d`")

        B = len(arg)
        B_re = required_length
        assert (
            B == B_re
        ), f"arg={arg} has a length of B={B} which isn't the required length={B_re}"

    order = range(sys.num_links())
    q_idx, qd_idx = 0, 0
    q_idxs, qd_idxs = {}, {}
    for link_idx, link_type in zip(order, sys.link_types):
        # build map from
        # link-idx -> q_idx
        # link-idx -> qd_idx
        q_idxs[link_idx] = slice(q_idx, q_idx + Q_WIDTHS[link_type])
        qd_idxs[link_idx] = slice(qd_idx, qd_idx + QD_WIDTHS[link_type])
        q_idx += Q_WIDTHS[link_type]
        qd_idx += QD_WIDTHS[link_type]

    idx_map = {
        "l": lambda link_idx: link_idx,
        "q": lambda link_idx: q_idxs[link_idx],
        "d": lambda link_idx: qd_idxs[link_idx],
    }

    if reverse:
        order = range(sys.num_links() - 1, -1, -1)

    y, ys = None, []
    for link_idx in order:
        args_link = [arg[idx_map[t](link_idx)] for arg, t in zip(args, in_types)]
        y = f(y, idx_map, *args_link)
        ys.append(y)

    if reverse:
        ys.reverse()

    ys = tu.tree_batch(ys, backend="jax")
    return ys


@struct.dataclass
class State(_Base):
    """The static and dynamic state of a system in minimal and maximal coordinates.
    Use `.create()` to create this object.

    Args:
        q (jax.Array): System state in minimal coordinates (equals `sys.q_size()`)
        qd (jax.Array): System velocity in minimal coordinates (equals `sys.qd_size()`)
        x: (Transform): Maximal coordinates of all links. From epsilon-to-link.
    """

    q: jax.Array
    qd: jax.Array
    x: Transform

    @classmethod
    def create(
        cls,
        sys: System,
        q: Optional[jax.Array] = None,
        qd: Optional[jax.Array] = None,
        x: Optional[Transform] = None,
        key: Optional[jax.Array] = None,
        custom_joints: dict[str, Callable] = {},
    ):
        """Create state of system.

        Args:
            sys (System): The system for which to create a state.
            q (jax.Array, optional): The joint values of the system. Defaults to None.
            Which then defaults to zeros.
            qd (jax.Array, optional): The joint velocities of the system.
            Defaults to None. Which then defaults to zeros.

        Returns:
            (State): Create State object.
        """
        if key is not None:
            assert q is None
            q = jax.random.normal(key, shape=(sys.q_size(),))
            q = sys.coordinate_vector_to_q(q, custom_joints)
        elif q is None:
            q = jnp.zeros((sys.q_size(),))

            # free, cor, spherical joints are not zeros but have unit quaternions
            def replace_by_unit_quat(_, idx_map, link_typ, link_idx):
                nonlocal q

                if sys.joint_type_is_free_or_cor_or_spherical(link_typ):
                    q_idxs_link = idx_map["q"](link_idx)
                    q = q.at[q_idxs_link.start].set(1.0)

            sys.scan(
                replace_by_unit_quat,
                "ll",
                sys.link_types,
                list(range(sys.num_links())),
            )
        else:
            pass

        if qd is None:
            qd = jnp.zeros((sys.qd_size(),))

        if x is None:
            x = Transform.zero((sys.num_links(),))

        return cls(q, qd, x)
