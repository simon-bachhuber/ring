import numpy as np
from ring.base import Color
from vispy.geometry.meshdata import MeshData
from vispy.scene.visuals import create_visual_node
from vispy.scene.visuals import Mesh
from vispy.visuals import CompoundVisual
from vispy.visuals import SphereVisual as _SphereVisual
from vispy.visuals import TubeVisual

# vertex density per unit length
_vectices_per_unit_length = 10

_default_color = (1, 0.8, 0.7, 1)
_default_edge_color = "black"


class DoubleMeshVisual(CompoundVisual):
    _lines: Mesh
    _faces: Mesh

    def __init__(
        self, verts, edges, faces, *, color: Color = None, edge_color: Color = None
    ):
        if color is None and edge_color is None:
            color = _default_color

        if color is not None:
            self._faces = Mesh(verts, faces, color=color, shading=None)
            self.light_dir = np.array([0, -1, 0])
        else:
            self._faces = Mesh()

        if edge_color is not None:
            self._edges = Mesh(verts, edges, color=edge_color, mode="lines")
        else:
            self._edges = Mesh()

        super().__init__([self._faces, self._edges])
        self._faces.set_gl_state(
            polygon_offset_fill=True, polygon_offset=(1, 1), depth_test=True
        )


class SphereVisual(_SphereVisual):
    def __init__(self, radius: float, color: Color = None, edge_color: Color = None):
        if color is None and edge_color is None:
            color = _default_color

        radius = float(radius)

        num_rows = max(int(np.pi * radius * _vectices_per_unit_length), 10)
        num_cols = max(int(2 * np.pi * radius * _vectices_per_unit_length), 20)

        super().__init__(
            radius,
            color=color,
            edge_color=edge_color,
            rows=num_rows,
            cols=num_cols,
            method="latitude",
            shading="smooth",
        )


Sphere = create_visual_node(SphereVisual)


def box_mesh(
    dim_x: float, dim_y: float, dim_z: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    verts = np.array(
        [
            (-dim_x, -dim_y, -dim_z),
            (dim_x, -dim_y, -dim_z),
            (-dim_x, dim_y, -dim_z),
            (dim_x, dim_y, -dim_z),
            (-dim_x, -dim_y, dim_z),
            (dim_x, -dim_y, dim_z),
            (-dim_x, dim_y, dim_z),
            (dim_x, dim_y, dim_z),
        ],
        dtype=np.float32,
    )

    verts /= 2

    edges = np.array(
        [
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        ],
        dtype=np.uint32,
    )

    faces = np.array(
        [
            (0, 1, 2),
            (1, 2, 3),
            (0, 1, 4),
            (1, 4, 5),
            (0, 2, 4),
            (2, 4, 6),
            (1, 3, 5),
            (3, 5, 7),
            (2, 3, 6),
            (3, 6, 7),
            (4, 5, 6),
            (5, 6, 7),
        ]
    )

    return verts, edges, faces


class BoxVisual(DoubleMeshVisual):
    # NOTE: need a custom BoxVisual class, since vispy.scene.visuals.Box does not
    # support shading

    def __init__(
        self,
        dim_x: float,
        dim_y: float,
        dim_z: float,
        *,
        color: Color = None,
        edge_color: Color = None
    ):
        if color is None:
            color = _default_color

        if edge_color is None:
            edge_color = _default_edge_color

        dim_x = float(dim_x)
        dim_y = float(dim_y)
        dim_z = float(dim_z)

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z

        verts, edges, faces = box_mesh(dim_x, dim_y, dim_z)

        super().__init__(verts, edges, faces, color=color, edge_color=edge_color)


Box = create_visual_node(BoxVisual)


class CylinderVisual(TubeVisual):
    def __init__(
        self,
        radius: float,
        length: float,
        *,
        color: Color = None,
        edge_color: Color = None
    ):
        if color is None and edge_color is None:
            color = _default_color

        radius = float(radius)
        length = float(length)

        num_length_points = 10 * max(int(length * _vectices_per_unit_length), 10)
        num_radial_points = max(int(2 * np.pi * radius * _vectices_per_unit_length), 20)

        points = np.zeros((num_length_points, 3))
        points[:, 0] = np.linspace(-length / 2, length / 2, num_length_points)

        self.radius = radius
        self.length = length

        super().__init__(
            points,
            radius,
            tube_points=num_radial_points,
            closed=True,
            color=color,
            shading="smooth",
        )


Cylinder = create_visual_node(CylinderVisual)


def capsule_mesh(radius: float, length: float, offset: bool = True) -> MeshData:
    if length < 2 * radius:
        raise ValueError("length must be at least 2 * radius")

    # number of cap vertices in x direction
    num_sphere_rows = max(int(radius * _vectices_per_unit_length), 10)

    # length without caps
    cyl_length = length - 2 * radius
    # number of cylinder vertices in x direction
    num_cyl_rows = max(int(cyl_length * _vectices_per_unit_length), 10)

    num_total_rows = 2 * num_sphere_rows + num_cyl_rows

    # number of radial vertices
    num_cols = max(int(2 * np.pi * radius * _vectices_per_unit_length), 20)

    verts = np.empty((num_total_rows, num_cols, 3), dtype=np.float32)

    # polar angle
    theta_top = np.linspace(0.0, np.pi / 2, num_sphere_rows)
    theta_bottom = np.linspace(np.pi / 2, np.pi, num_sphere_rows)

    # fill in x coordinate
    verts[:num_sphere_rows, :, 0] = radius * np.cos(theta_top[:, None]) + cyl_length / 2

    verts[num_sphere_rows:-num_sphere_rows, :, 0] = np.linspace(
        -cyl_length / 2, cyl_length / 2, num_cyl_rows
    )[::-1, None]

    verts[-num_sphere_rows:, :, 0] = (
        radius * np.cos(theta_bottom[:, None]) - cyl_length / 2
    )

    # azimuth angle
    phi = (np.linspace(0, 2 * np.pi, num_cols))[None, :]

    if offset:
        # rotate each row by 1/2 column
        phi = phi + (np.pi / num_cols) * np.arange(num_total_rows)[:, None]

    # y and z coordinates
    verts[..., 1] = radius * np.cos(phi)
    verts[..., 2] = radius * np.sin(phi)

    # for caps: bend inwards to close
    verts[:num_sphere_rows, :, 1:3] *= np.sin(theta_top[:, None, None])
    verts[-num_sphere_rows:, :, 1:3] *= np.sin(theta_bottom[:, None, None])

    verts = verts.reshape(-1, 3)

    # compute faces
    faces = np.empty(((num_total_rows - 1) * num_cols * 2, 3), dtype=np.uint32)

    rowtemplate1 = (
        (np.arange(num_cols).reshape(num_cols, 1) + np.array([[0, 1, 0]])) % num_cols
    ) + np.array([[num_cols, 0, 0]])

    rowtemplate2 = (
        (np.arange(num_cols).reshape(num_cols, 1) + np.array([[1, 1, 0]])) % num_cols
    ) + np.array([[num_cols, 0, num_cols]])

    for row in range(num_total_rows - 1):
        start = row * num_cols * 2

        faces[start : start + num_cols] = rowtemplate1 + row * num_cols
        faces[start + num_cols : start + (num_cols * 2)] = rowtemplate2 + row * num_cols

    mesh = MeshData(vertices=verts, faces=faces)

    return mesh.get_vertices(), mesh.get_edges(), mesh.get_faces()


class CapsuleVisual(DoubleMeshVisual):
    def __init__(
        self,
        radius: float,
        length: float,
        *,
        color: Color = None,
        edge_color: Color = None
    ):
        radius = float(radius)
        length = float(length)

        self.radius = radius
        self.length = length

        verts, edges, faces = capsule_mesh(radius, length)

        super().__init__(verts, edges, faces, color=color, edge_color=edge_color)


Capsule = create_visual_node(CapsuleVisual)
