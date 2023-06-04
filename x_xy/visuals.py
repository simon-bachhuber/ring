from vispy.visuals import MeshVisual
from vispy.geometry.meshdata import MeshData
from vispy.scene.visuals import create_visual_node

import numpy as np


def box_mesh(dim_x: float, dim_y: float, dim_z: float) -> tuple[np.ndarray, np.ndarray]:
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

    faces = np.array(
        [
            (0, 1, 1),
            (0, 2, 2),
            (0, 4, 4),
            (1, 3, 3),
            (1, 5, 5),
            (2, 3, 3),
            (2, 6, 6),
            (3, 7, 7),
            (4, 5, 5),
            (4, 6, 6),
            (5, 7, 7),
            (6, 7, 7),
        ],
        dtype=np.uint32,
    )

    mesh = MeshData(vertices=verts, faces=faces)

    return mesh.get_vertices(), mesh.get_edges()

    # return verts, faces


class BoxVisual(MeshVisual):
    def __init__(self, dim_x: float, dim_y: float, dim_z: float, **kwargs):
        dim_x = float(dim_x)
        dim_y = float(dim_y)
        dim_z = float(dim_z)

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z

        verts, faces = box_mesh(dim_x, dim_y, dim_z)

        super().__init__(verts, faces, mode="lines", **kwargs)


Box = create_visual_node(BoxVisual)


def sphere_ico_mesh(radius: float) -> MeshData:
    # golden ratio
    t = (1.0 + np.sqrt(5.0)) / 2.0

    subdivisions = max(int(radius), 1)

    # semipositive vertices of a icosahedron
    verts = [
        (-1, t, 0),
        (1, t, 0),
        (-1, -t, 0),
        (1, -t, 0),
        (0, -1, t),
        (0, 1, t),
        (0, -1, -t),
        (0, 1, -t),
        (t, 0, -1),
        (t, 0, 1),
        (-t, 0, -1),
        (-t, 0, 1),
    ]

    # faces of the icosahedron
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]

    def midpoint(v1, v2):
        return ((v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2, (v1[2] + v2[2]) / 2)

    # subdivision
    for _ in range(subdivisions):
        for idx in range(len(faces)):
            i, j, k = faces[idx]
            a, b, c = verts[i], verts[j], verts[k]
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            verts += [ab, bc, ca]
            ij, jk, ki = len(verts) - 3, len(verts) - 2, len(verts) - 1
            faces.append([i, ij, ki])
            faces.append([ij, j, jk])
            faces.append([ki, jk, k])
            faces[idx] = [jk, ki, ij]

    verts = np.array(verts, dtype=np.float32)
    faces = np.array(faces, dtype=np.uint32)

    # make each vertex to lie on the sphere
    lengths = np.sqrt((verts * verts).sum(axis=1))
    verts /= lengths[:, np.newaxis] / float(radius)

    mesh = MeshData(vertices=verts, faces=faces)

    return mesh.get_vertices(), mesh.get_edges()


class SphereVisual(MeshVisual):
    def __init__(self, radius: float, **kwargs):
        radius = float(radius)

        self.radius = radius

        verts, faces = sphere_ico_mesh(radius)

        super().__init__(verts, faces, mode="lines", **kwargs)


Sphere = create_visual_node(SphereVisual)


def cylinder_mesh(
    radius: float,
    length: float,
) -> tuple[np.ndarray, np.ndarray]:
    cols = 8
    rows = max(4 * int(length), 3) + 2

    verts = np.empty((rows, cols, 3), dtype=np.float32)

    x = np.linspace(-length / 2, length / 2, rows - 2)[:, None]

    verts[0, :, 0] = -length / 2
    verts[1:-1, :, 0] = x
    verts[-1, :, 0] = length / 2

    th = np.linspace(0, 2 * np.pi, cols)[None, :]

    # rotate each row by 1/2 column
    th = th + (np.pi / cols) * np.arange(rows - 2)[:, None]

    verts[1:-1, :, 1] = radius * np.cos(th)
    verts[1:-1, :, 2] = radius * np.sin(th)

    verts[0, :, 1:3] = 0
    verts[-1, :, 1:3] = 0

    verts = verts.reshape(-1, 3)

    # compute faces
    faces = np.empty(((rows - 1) * cols * 2, 3), dtype=np.uint32)

    rowtemplate1 = (
        (np.arange(cols).reshape(cols, 1) + np.array([[0, 1, 0]])) % cols
    ) + np.array([[cols, 0, 0]])
    rowtemplate2 = (
        (np.arange(cols).reshape(cols, 1) + np.array([[1, 1, 0]])) % cols
    ) + np.array([[cols, 0, cols]])

    for row in range(rows - 1):
        start = row * cols * 2

        faces[start : start + cols] = rowtemplate1 + row * cols
        faces[start + cols : start + (cols * 2)] = rowtemplate2 + row * cols

    mesh = MeshData(vertices=verts, faces=faces)

    return mesh.get_vertices(), mesh.get_edges()


class CylinderVisual(MeshVisual):
    def __init__(self, radius: float, length: float, **kwargs):
        radius = float(radius)
        length = float(length)

        self.radius = radius
        self.length = length

        verts, faces = cylinder_mesh(radius, length)

        super().__init__(verts, faces, mode="lines", **kwargs)


Cylinder = create_visual_node(CylinderVisual)


def capsule_mesh(radius: float, length: float, offset: bool = True) -> MeshData:
    if length < 2 * radius:
        raise ValueError("length must be at least 2 * radius")

    sphere_rows = max(8 * int(radius), 5)

    cols = 8

    cyl_length = length - 2 * radius
    cyl_rows = max(4 * int(cyl_length), 3)

    total_rows = 2 * sphere_rows + cyl_rows

    verts = np.empty((total_rows, cols, 3), dtype=np.float32)

    # compute vertices
    phi = np.linspace(0.0, np.pi, 2 * sphere_rows)

    verts[:sphere_rows, :, 0] = (
        radius * np.cos(phi[:sphere_rows, None]) + cyl_length / 2
    )
    verts[sphere_rows:-sphere_rows, :, 0] = np.linspace(
        -cyl_length / 2, cyl_length / 2, cyl_rows
    )[::-1, None]
    verts[-sphere_rows:, :, 0] = (
        radius * np.cos(phi[-sphere_rows:, None]) - cyl_length / 2
    )

    # th = (np.arange(cols) * 2 * np.pi / cols).reshape(1, cols)
    th = (np.linspace(0, 2 * np.pi, cols))[None, :]

    if offset:
        # rotate each row by 1/2 column
        th = th + (np.pi / cols) * np.arange(total_rows)[:, None]

    # s = np.empty((2 * rows + cyl_rows, 1))

    # s[:rows, 0] = radius * np.sin(phi[:rows])
    # s[rows:-rows, 0] = radius
    # s[-rows:, 0] = radius * np.sin(phi[-rows:])

    verts[..., 1] = radius * np.cos(th)
    verts[..., 2] = radius * np.sin(th)

    verts[:sphere_rows, :, 1:3] *= np.sin(phi[:sphere_rows, None, None])
    verts[-sphere_rows:, :, 1:3] *= np.sin(phi[-sphere_rows:, None, None])

    # remove redundant vertices from top and bottom
    verts = verts.reshape(-1, 3)  # [cols - 1 : -(cols - 1)]

    # compute faces
    faces = np.empty(((total_rows - 1) * cols * 2, 3), dtype=np.uint32)

    rowtemplate1 = (
        (np.arange(cols).reshape(cols, 1) + np.array([[0, 1, 0]])) % cols
    ) + np.array([[cols, 0, 0]])

    rowtemplate2 = (
        (np.arange(cols).reshape(cols, 1) + np.array([[1, 1, 0]])) % cols
    ) + np.array([[cols, 0, cols]])

    for row in range(total_rows - 1):
        start = row * cols * 2

        faces[start : start + cols] = rowtemplate1 + row * cols
        faces[start + cols : start + (cols * 2)] = rowtemplate2 + row * cols

    # cut off zero-area triangles at top
    # faces = faces[cols:]

    # adjust for redundant vertices that were removed from top
    # vmin = cols - 1
    # faces[faces < vmin] = vmin
    # faces -= vmin
    # vmax = verts.shape[0] - 1
    # faces[faces > vmax] = vmax

    mesh = MeshData(vertices=verts, faces=faces)

    return mesh.get_vertices(), mesh.get_edges()


class CapsuleVisual(MeshVisual):
    def __init__(self, radius: float, length: float, **kwargs):
        radius = float(radius)
        length = float(length)

        self.radius = radius
        self.length = length

        # rows = kwargs.pop("rows", 5)
        # cols = kwargs.pop("cols", 8)

        verts, faces = capsule_mesh(radius, length)

        # self.capsule = MeshVisual(
        #     vertices=verts, faces=faces, color=color, mode="lines"
        # )

        # self.sphere1, self.sphere2, self.cylinder
        super().__init__(verts, faces, mode="lines", **kwargs)


Capsule = create_visual_node(CapsuleVisual)
