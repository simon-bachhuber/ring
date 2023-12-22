from typing import Any, Optional, Sequence, Union

from flax import struct
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import numpy as np
import tree_utils as tu

from x_xy import maths

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

    def shape(self, axis=0) -> int:
        return tu.tree_shape(self, axis)


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
    def create(cls, mass: Vector, transform: Transform, it_3x3: jnp.ndarray):
        """Construct spatial inertia of an object with mass `mass` located and aligned
        with a coordinate system that is given by `transform` where `transform` is from
        parent to local geometry coordinates.
        """
        it_3x3 = maths.rotate_matrix(it_3x3, maths.quat_inv(transform.rot))
        it_3x3 = maths.spatial.mcI(mass, transform.pos, it_3x3)[:3, :3]
        h = mass * transform.pos
        return cls(it_3x3, h, mass)

    @classmethod
    def zero(cls, shape=()) -> "Inertia":
        it_shape_3x3 = jnp.zeros(shape + (3, 3))
        h = jnp.zeros(shape + (3,))
        mass = jnp.zeros(shape + (1,))
        return cls(it_shape_3x3, h, mass)

    def as_matrix(self):
        hcross = maths.spatial.cross(self.h)
        return maths.spatial.quadrants(
            self.it_3x3, hcross, -hcross, self.mass * jnp.eye(3)
        )


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


Q_WIDTHS = {
    "free": 7,
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

    def num_links(self) -> int:
        return len(self.link_parents)

    def q_size(self) -> int:
        return sum([Q_WIDTHS[typ] for typ in self.link_types])

    def qd_size(self) -> int:
        return sum([QD_WIDTHS[typ] for typ in self.link_types])

    def name_to_idx(self, name: str) -> int:
        return self.link_names.index(name)

    def idx_to_name(self, idx: int) -> str:
        assert idx >= 0, "Worldbody index has no name."
        return self.link_names[idx]

    def idx_map(self, type: str) -> dict:
        "type: is either `l` or `q` or `d`"
        from x_xy import scan_sys

        dict_int_slices = {}

        def f(_, idx_map, name: str, link_idx: int):
            dict_int_slices[name] = idx_map[type](link_idx)

        scan_sys(self, f, "ll", self.link_names, list(range(self.num_links())))

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

    @staticmethod
    def deep_equal(a, b):
        if type(a) is not type(b):
            return False
        if isinstance(a, _Base):
            return System.deep_equal(a.__dict__, b.__dict__)
        if isinstance(a, dict):
            if a.keys() != b.keys():
                return False
            return all(System.deep_equal(a[k], b[k]) for k in a.keys())
        if isinstance(a, (list, tuple)):
            if len(a) != len(b):
                return False
            return all(System.deep_equal(a[i], b[i]) for i in range(len(a)))
        if isinstance(a, (np.ndarray, jnp.ndarray, jax.Array)):
            return jnp.array_equal(a, b)
        return a == b

    def _replace_free_with_cor(self) -> "System":
        # check that
        # - all free joints connect to -1
        # - all joints connecting to -1 are free joints
        for i, p in enumerate(self.link_parents):
            link_type = self.link_types[i]
            if (p == -1 and link_type != "free") or (link_type == "free" and p != -1):
                raise InvalidSystemError(
                    f"link={self.idx_to_name(i)}, parent={self.idx_to_name(p)},"
                    f" joint={link_type}"
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
    ):
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

        return _update_sys_if_replace_joint_type(self, logic_unfreeze_to_spherical)

    def findall_imus(self) -> list[str]:
        return [name for name in self.link_names if name[:3] == "imu"]

    def findall_segments(self) -> list[str]:
        imus = self.findall_imus()
        return [name for name in self.link_names if name not in imus]


def _update_sys_if_replace_joint_type(sys: System, logic) -> System:
    lt, la, ld, ls, lz = [], [], [], [], []

    def f(_, __, name, olt, ola, old, ols, olz):
        nlt, nla, nld, nls, nlz = logic(name, olt, ola, old, ols, olz)

        lt.append(nlt)
        la.append(nla)
        ld.append(nld)
        ls.append(nls)
        lz.append(nlz)

    from x_xy import scan_sys

    scan_sys(
        sys,
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

    from x_xy.io import parse_system

    # parse system such that it checks if all joint types have the
    # correct dimensionality of damping / stiffness / zeropoint / armature
    return parse_system(sys)


class InvalidSystemError(Exception):
    pass


@struct.dataclass
class State(_Base):
    """The static and dynamic state of a system in minimal and maximal coordinates.
    Use `.create()` to create this object.

    Args:
        q (jax.Array): System state in minimal coordinates (equals `sys.q_size()`)
        qd (jax.Array): System velocity in minimal coordinates (equals `sys.qd_size()`)
        x: (Transform): Maximal coordinates of all links. From epsilon-to-link.
        mass_mat_inv (jax.Array): Inverse of the mass matrix. Internal usage.
    """

    q: jax.Array
    qd: jax.Array
    x: Transform
    mass_mat_inv: jax.Array

    @classmethod
    def create(
        cls, sys: System, q: Optional[jax.Array] = None, qd: Optional[jax.Array] = None
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
        # to avoid circular imports
        from x_xy import scan_sys

        if q is None:
            q = jnp.zeros((sys.q_size(),))

            # free, cor, spherical joints are not zeros but have unit quaternions
            def replace_by_unit_quat(_, idx_map, link_typ, link_idx):
                nonlocal q

                if link_typ in ["free", "cor", "spherical"]:
                    q_idxs_link = idx_map["q"](link_idx)
                    q = q.at[q_idxs_link.start].set(1.0)

            scan_sys(
                sys,
                replace_by_unit_quat,
                "ll",
                sys.link_types,
                list(range(sys.num_links())),
            )

        if qd is None:
            qd = jnp.zeros((sys.qd_size(),))
        x = Transform.zero((sys.num_links(),))
        return cls(q, qd, x, jnp.diag(jnp.ones((sys.qd_size(),))))
