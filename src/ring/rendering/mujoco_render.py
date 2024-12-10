from typing import Optional, Sequence

import mujoco
import numpy as np

from ring import base
from ring import maths

_skybox = """<texture name="skybox" type="skybox" builtin="gradient" rgb1=".4 .6 .8" rgb2="0 0 0" width="800" height="800" mark="random" markrgb="1 1 1"/>"""  # noqa: E501
_skybox_white = """<texture name="skybox" type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800" mark="random" markrgb="1 1 1"/>"""  # noqa: E501


def _floor(floor_z: float) -> str:
    return f"""<geom name="floor" pos="0 0 {floor_z}" size="0 0 1" type="plane" material="matplane" mass="0"/>"""  # noqa: E501


def _build_model_of_geoms(
    geoms: list[base.Geometry],
    cameras: dict[int, Sequence[str]],
    lights: dict[int, Sequence[str]],
    floor: bool,
    floor_z: float,
    stars: bool,
    debug: bool,
) -> mujoco.MjModel:
    # sort in ascending order, this shouldn't be required as it is already done by
    geoms = geoms.copy()
    geoms.sort(key=lambda ele: ele.link_idx)

    # range of required link_indices to which geoms attach
    unique_parents = set([geom.link_idx for geom in geoms])

    # throw error if you attached a camera or light to a body that has no geoms
    inside_worldbody_cameras = ""
    for camera_parent in cameras:
        if -1 not in unique_parents:
            if camera_parent == -1:
                for camera_str in cameras[camera_parent]:
                    inside_worldbody_cameras += camera_str
                continue

        assert (
            camera_parent in unique_parents
        ), f"Camera parent {camera_parent} not in {unique_parents}"

    inside_worldbody_lights = ""
    for light_parent in lights:
        if -1 not in unique_parents:
            if light_parent == -1:
                for light_str in lights[light_parent]:
                    inside_worldbody_lights += light_str
                continue

        assert (
            light_parent in unique_parents
        ), f"Light parent {light_parent} not in {unique_parents}"

    # group together all geoms in each link
    grouped_geoms = dict(
        zip(unique_parents, [list() for _ in range(len(unique_parents))])
    )
    parent = -1
    for geom in geoms:
        while geom.link_idx != parent:
            parent += 1
        grouped_geoms[parent].append(geom)

    inside_worldbody = ""
    for parent, geoms in grouped_geoms.items():
        find = lambda dic: dic[parent] if parent in dic else []
        inside_worldbody += _xml_str_one_body(
            parent, geoms, find(cameras), find(lights)
        )

    parents_noworld = unique_parents - set([-1])
    targetbody = min(parents_noworld) if len(parents_noworld) > 0 else -1
    xml_str = f""" # noqa: E501
<mujoco>
  <asset>
    <texture name="texplane" type="2d" builtin="checker" rgb1=".25 .25 .25" rgb2=".3 .3 .3" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
    <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="2 2" reflectance="0.2"/>
    {_skybox if stars else ''}
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
  </asset>

  <visual>
    <headlight ambient=".4 .4 .4" diffuse=".8 .8 .8" specular="0.1 0.1 0.1"/>
    <map znear=".01"/>
    <quality shadowsize="8192"/>
    <global offwidth="3840" offheight="2160"/>
  </visual>

<worldbody>
<camera pos="0 -1 1" name="trackcom" mode="trackcom"/>
<camera pos="0 -1 1" name="target" mode="targetbodycom" target="{targetbody}"/>
<camera pos="0 -3 3" name="targetfar" mode="targetbodycom" target="{targetbody}"/>
<camera pos="0 -5 5" name="targetFar" mode="targetbodycom" target="{targetbody}"/>
{_floor(floor_z) if floor else ''}
{inside_worldbody_cameras}
{inside_worldbody_lights}
{inside_worldbody}
</worldbody>
</mujoco>
"""
    if debug:
        print("Mujoco xml string: ", xml_str)

    return mujoco.MjModel.from_xml_string(xml_str)


def _xml_str_one_body(
    body_number: int, geoms: list[base.Geometry], cameras: list[str], lights: list[str]
) -> str:
    inside_body_geoms = ""
    for geom in geoms:
        inside_body_geoms += _xml_str_one_geom(geom)

    inside_body_cameras = ""
    for camera in cameras:
        inside_body_cameras += camera  # + "\n"

    inside_body_lights = ""
    for light in lights:
        inside_body_lights += light  # + "\n"

    return f"""
<body name="{body_number}" mocap="true">
{inside_body_cameras}
{inside_body_lights}
{inside_body_geoms}
</body>
"""


def _xml_str_one_geom(geom: base.Geometry) -> str:
    rgba = f'rgba="{_array_to_str(geom.color)}"'

    if isinstance(geom, base.Box):
        type_size = f'type="box" size="{_array_to_str([geom.dim_x / 2, geom.dim_y / 2, geom.dim_z / 2])}"'  # noqa: E501
    elif isinstance(geom, base.Sphere):
        type_size = f'type="sphere" size="{_array_to_str([geom.radius])}"'
    elif isinstance(geom, base.Capsule):
        type_size = (
            f'type="capsule" size="{_array_to_str([geom.radius, geom.length / 2])}"'
        )
    elif isinstance(geom, base.Cylinder):
        type_size = (
            f'type="cylinder" size="{_array_to_str([geom.radius, geom.length / 2])}"'
        )
    else:
        raise NotImplementedError

    rot, pos = maths.quat_inv(geom.transform.rot), geom.transform.pos
    rot, pos = f'pos="{_array_to_str(pos)}"', f'quat="{_array_to_str(rot)}"'
    return f"<geom {type_size} {rgba} {rot} {pos}/>"


def _array_to_str(arr: Sequence[float]) -> str:
    # TODO; remove round & truncation
    return "".join(["{:.4f} ".format(np.round(value, 4)) for value in arr])[:-1]


_default_lights = {-1: '<light pos="0 0 4" dir="0 0 -1"/>'}


class MujocoScene:
    def __init__(
        self,
        height: int = 240,
        width: int = 320,
        add_cameras: dict[int, str | Sequence[str]] = {},
        add_lights: dict[int, str | Sequence[str]] = _default_lights,
        show_stars: bool = True,
        show_floor: bool = True,
        floor_z: float = -0.84,
        debug: bool = False,
    ) -> None:
        self.debug = debug
        self.height, self.width = height, width

        def to_list(dic: dict):
            for k, v in dic.items():
                if isinstance(v, str):
                    dic[k] = [v]
            return dic

        self.add_cameras, self.add_lights = to_list(add_cameras), to_list(add_lights)
        self.show_stars = show_stars
        self.show_floor = show_floor
        self.floor_z = floor_z

    def init(self, geoms: list[base.Geometry]):
        self._parent_ids = list(set([geom.link_idx for geom in geoms]))
        self._model = _build_model_of_geoms(
            geoms,
            self.add_cameras,
            self.add_lights,
            floor=self.show_floor,
            floor_z=self.floor_z,
            stars=self.show_stars,
            debug=self.debug,
        )
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, self.height, self.width)

    def update(self, x: base.Transform):
        rot, pos = maths.quat_inv(x.rot), x.pos
        for parent_id in self._parent_ids:
            if parent_id == -1:
                continue

            # body name is just the str(parent_id)
            # squeeze reduces shape (1,) to () which removes a warning
            mocap_id = int(np.squeeze(self._model.body(str(parent_id)).mocapid))

            if self.debug:
                print(f"link_idx: {parent_id}, mocap_id: {mocap_id}")

            mocap_pos = pos[parent_id]
            mocap_quat = rot[parent_id]
            self._data.mocap_pos[mocap_id] = mocap_pos
            self._data.mocap_quat[mocap_id] = mocap_quat

        if self.debug:
            print("mocap_pos: ", self._data.mocap_pos)
            print("mocap_quat: ", self._data.mocap_quat)

        mujoco.mj_forward(self._model, self._data)

    def render(self, camera: Optional[str] = None):
        self._renderer.update_scene(self._data, camera=-1 if camera is None else camera)
        return self._renderer.render()
