from typing import Tuple, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from ring import base

T = TypeVar("T")
ATTR = dict

default_quat = jnp.array([1.0, 0, 0, 0])
default_pos = jnp.zeros((3,))
default_damping = lambda qd_size, **_: jnp.zeros((qd_size,))
default_armature = lambda qd_size, **_: jnp.zeros((qd_size,))
default_stiffness = lambda qd_size, **_: jnp.zeros((qd_size,))


def default_zeropoint(q_size, link_typ: str, **_):
    zeropoint = jnp.zeros((q_size))
    if base.System.joint_type_is_free_or_cor_or_spherical(link_typ):
        # zeropoint then is unit quaternion and not zeros
        zeropoint = zeropoint.at[0].set(1.0)
    return zeropoint


default_fns = {
    "damping": default_damping,
    "armature": default_armature,
    "spring_stiff": default_stiffness,
    "spring_zero": default_zeropoint,
}


class AbsDampArmaStiffZero:
    @staticmethod
    def from_xml(attr: ATTR, q_size: int, qd_size: int, link_typ: str) -> list:
        return [
            jnp.atleast_1d(
                attr.get(
                    key,
                    default_fns[key](q_size=q_size, qd_size=qd_size, link_typ=link_typ),
                )
            )
            for key in ["damping", "armature", "spring_stiff", "spring_zero"]
        ]

    @staticmethod
    def to_xml(
        element: T,
        damping: jax.Array,
        armature: jax.Array,
        stiffness: jax.Array,
        zeropoint: jax.Array,
        q_size: int,
        qd_size: int,
        link_typ: str,
    ):
        for key, arr in zip(
            ["damping", "armature", "spring_stiff", "spring_zero"],
            [damping, armature, stiffness, zeropoint],
        ):
            if not _arr_equal(
                arr, default_fns[key](q_size=q_size, qd_size=qd_size, link_typ=link_typ)
            ):
                element.set(key, _to_str(arr))


class AbsMaxCoordOMC:
    @staticmethod
    def from_xml(attr: ATTR) -> base.MaxCoordOMC:
        pos = attr.get("pos", default_pos)
        marker_number = int(attr.get("pos_marker"))
        cs_name = attr.get("name")
        return base.MaxCoordOMC(cs_name, marker_number, pos)

    @staticmethod
    def to_xml(element: T, max_coord_omc: base.MaxCoordOMC) -> None:
        if not _arr_equal(max_coord_omc.pos_marker_constant_offset, default_pos):
            element.set("pos", _to_str(max_coord_omc.pos_marker_constant_offset))
        element.set("name", max_coord_omc.coordinate_system_name)
        element.set("pos_marker", _to_str(max_coord_omc.pos_marker_number))


class AbsTrans:
    @staticmethod
    def from_xml(attr: ATTR) -> base.Transform:
        pos = attr.get("pos", default_pos)
        rot = _get_rotation(attr)
        return base.Transform(pos, rot)

    @staticmethod
    def to_xml(element: T, t: base.Transform) -> None:
        if not _arr_equal(t.pos, default_pos):
            element.set("pos", _to_str(t.pos))
        if not _arr_equal(t.rot, default_quat):
            element.set("quat", _to_str(t.rot))


class AbsPosMinMax:
    @staticmethod
    def from_xml(attr: ATTR, pos: jax.Array) -> Tuple[jax.Array, jax.Array]:
        pos_min = attr.get("pos_min", None)
        pos_max = attr.get("pos_max", None)
        assert (pos_min is None and pos_max is None) or (
            pos_min is not None and pos_max is not None
        ), (
            f"In link {attr.get('name', 'None')} found only one of `pos_min` "
            "and `pos_max`, but requires either both or none"
        )
        if pos_min is not None:
            assert not _arr_equal(
                pos_min, pos_max
            ), f"In link {attr.get('name', 'None')} "
            " both `pos_min` and `pos_max` are identical, use `pos` instead."

        if pos_min is None:
            pos_min = pos_max = pos
        return pos_min, pos_max

    @staticmethod
    def to_xml(element: T, pos_min: jax.Array, pos_max: jax.Array):
        if _arr_equal(pos_min, pos_max):
            return

        element.set("pos_min", _to_str(pos_min))
        element.set("pos_max", _to_str(pos_max))


def _from_xml_geom_attr_processing(geom_attr: ATTR):
    "Common processing used by all geometries"

    mass = geom_attr["mass"]
    trafo = AbsTrans.from_xml(geom_attr)

    # convert arrays to tuple[float], because of `struct.field(False)`
    # Otherwise jitted functions with `sys` input will error on second execution, since
    # it can't compare the two vispy_color arrays.

    color = geom_attr.get("color", None)
    if isinstance(color, (jax.Array, np.ndarray)):
        color = tuple(color.tolist())

    edge_color = geom_attr.get("edge_color", None)
    if isinstance(edge_color, (jax.Array, np.ndarray)):
        edge_color = tuple(edge_color.tolist())

    return mass, trafo, color, edge_color


def _to_xml_geom_processing(element: T, geom: base.Geometry) -> None:
    "Common processing used by all geometries"
    AbsTrans.to_xml(element, geom.transform)

    element.set("mass", _to_str(geom.mass))

    if geom.color is not None:
        element.set("color", _to_str(geom.color))

    if geom.edge_color is not None:
        element.set("edge_color", _to_str(geom.edge_color))

    element.set("type", geometry_to_xml_identifier[type(geom)])


class AbsGeomBox:
    xml_geom_type: str = "box"
    geometry: base.Geometry = base.Box

    @staticmethod
    def from_xml(geom_attr: ATTR, link_idx: int) -> base.Box:
        mass, trafo, color, edge_color = _from_xml_geom_attr_processing(geom_attr)
        dims = [geom_attr["dim"][i] for i in range(3)]
        assert all([dim > 0.0 for dim in dims]), "Negative box dimensions"
        return base.Box(mass, trafo, link_idx, color, edge_color, *dims)

    @staticmethod
    def to_xml(element: T, geom: base.Box) -> None:
        _to_xml_geom_processing(element, geom)
        dim = np.array([geom.dim_x, geom.dim_y, geom.dim_z])
        element.set("dim", _to_str(dim))


class AbsGeomSphere:
    xml_geom_type: str = "sphere"
    geometry: base.Geometry = base.Sphere

    @staticmethod
    def from_xml(geom_attr: ATTR, link_idx: int) -> base.Sphere:
        mass, trafo, color, edge_color = _from_xml_geom_attr_processing(geom_attr)
        radius = geom_attr["dim"].item()
        assert radius > 0.0, "Negative sphere radius"
        return base.Sphere(mass, trafo, link_idx, color, edge_color, radius)

    @staticmethod
    def to_xml(element: T, geom: base.Sphere) -> None:
        _to_xml_geom_processing(element, geom)
        dim = np.array([geom.radius])
        element.set("dim", _to_str(dim))


class AbsGeomCylinder:
    xml_geom_type: str = "cylinder"
    geometry: base.Geometry = base.Cylinder

    @staticmethod
    def from_xml(geom_attr: ATTR, link_idx: int) -> base.Cylinder:
        mass, trafo, color, edge_color = _from_xml_geom_attr_processing(geom_attr)
        dims = [geom_attr["dim"][i] for i in range(2)]
        assert all([dim > 0.0 for dim in dims]), "Negative cylinder dimensions"
        return base.Cylinder(mass, trafo, link_idx, color, edge_color, *dims)

    @staticmethod
    def to_xml(element: T, geom: base.Cylinder) -> None:
        _to_xml_geom_processing(element, geom)
        dim = np.array([geom.radius, geom.length])
        element.set("dim", _to_str(dim))


class AbsGeomCapsule:
    xml_geom_type: str = "capsule"
    geometry: base.Geometry = base.Capsule

    @staticmethod
    def from_xml(geom_attr: ATTR, link_idx: int) -> base.Capsule:
        mass, trafo, color, edge_color = _from_xml_geom_attr_processing(geom_attr)
        dims = [geom_attr["dim"][i] for i in range(2)]
        assert all([dim > 0.0 for dim in dims]), "Negative capsule dimensions"
        return base.Capsule(mass, trafo, link_idx, color, edge_color, *dims)

    @staticmethod
    def to_xml(element: T, geom: base.Capsule) -> None:
        _to_xml_geom_processing(element, geom)
        dim = np.array([geom.radius, geom.length])
        element.set("dim", _to_str(dim))


class AbsGeomXYZ:
    xml_geom_type: str = "xyz"
    geometry: base.Geometry = base.XYZ

    @staticmethod
    def from_xml(geom_attr: ATTR, link_idx: int) -> base.XYZ:
        if "dim" in geom_attr:
            dim = geom_attr["dim"]
        else:
            dim = 1.0

        assert dim > 0, "Negative xyz dimensions"
        return base.XYZ.create(link_idx, dim)

    @staticmethod
    def to_xml(element: T, geom: base.XYZ):
        element.set("type", geometry_to_xml_identifier[type(geom)])

        if geom.size != 1.0:
            element.set("dim", _to_str(geom.size))


_ags = [
    AbsGeomBox,
    AbsGeomSphere,
    AbsGeomCylinder,
    AbsGeomCapsule,
    AbsGeomXYZ,
]
geometry_to_xml_identifier = {ag.geometry: ag.xml_geom_type for ag in _ags}
xml_identifier_to_abstract = {ag.xml_geom_type: ag for ag in _ags}
geometry_to_abstract = {ag.geometry: ag for ag in _ags}


def _arr_equal(a, b):
    return np.all(np.array_equal(a, b))


def _get_rotation(attr: ATTR):
    rot = attr.get("quat", None)
    if rot is not None:
        assert "euler" not in attr, "Can't specify both `quat` and `euler` in xml"
    elif "euler" in attr:
        # we use zyx convention but angles are given
        # in x, y, z in the xml file
        # thus flip the order
        euler_xyz = jnp.deg2rad(attr["euler"])
        rot = base.maths.quat_euler(jnp.flip(euler_xyz), convention="zyx")
    else:
        rot = default_quat
    return rot


def _to_str(obj):
    if isinstance(obj, list) or isinstance(obj, tuple):
        if all([isinstance(ele, float) for ele in obj]):
            obj = np.array(obj)

    if isinstance(obj, (np.ndarray, jnp.ndarray)):
        if obj.ndim == 0:
            return str(obj)
        return " ".join([str(x) for x in obj])
    else:
        return str(obj)
