from typing import Sequence

import mujoco

from ..base import Box
from ..base import Capsule
from ..base import Cylinder
from ..base import Geometry
from ..base import Sphere
from ..base import Transform


def _build_model_of_geoms(geoms: list[Geometry]) -> mujoco.MjModel:
    # sort in ascending order, this shouldn't be required as it is already done by
    # parse_system; do it for good measure anyways
    geoms = geoms.copy()
    geoms.sort(key=lambda ele: ele.link_idx)

    # range of required link_indices to which geoms attach
    unique_parents = set([geom.link_idx for geom in geoms])

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
        inside_worldbody += _xml_str_one_body(parent, geoms)

    xml_str = f""" # noqa: E501
<mujoco>
  <asset>
    <texture name="texplane" type="2d" builtin="checker" rgb1=".25 .25 .25" rgb2=".3 .3 .3" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
    <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1=".4 .6 .8" rgb2="0 0 0" width="800" height="800" mark="random" markrgb="1 1 1"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
    <material name="self" rgba=".7 .5 .3 1"/>
    <material name="self_default" rgba=".7 .5 .3 1"/>
    <material name="self_highlight" rgba="0 .5 .3 1"/>
    <material name="effector" rgba=".7 .4 .2 1"/>
    <material name="effector_default" rgba=".7 .4 .2 1"/>
    <material name="effector_highlight" rgba="0 .5 .3 1"/>
    <material name="decoration" rgba=".3 .5 .7 1"/>
    <material name="eye" rgba="0 .2 1 1"/>
    <material name="target" rgba=".6 .3 .3 1"/>
    <material name="target_default" rgba=".6 .3 .3 1"/>
    <material name="target_highlight" rgba=".6 .3 .3 .4"/>
    <material name="site" rgba=".5 .5 .5 .3"/>
    <material name="red" rgba="0.8 0.2 0.2 1"/>
    <material name="green" rgba="0.2 0.8 0.2 1"/>
    <material name="blue" rgba="0.2 0.2 0.8 1"/>
    <material name="yellow" rgba="0.8 0.8 0.2 1"/>
    <material name="cyan" rgba="0.2 0.8 0.8 1"/>
    <material name="magenta" rgba="0.8 0.2 0.8 1"/>
    <material name="white" rgba="0.8 0.8 0.8 1"/>
    <material name="gray" rgba="0.5 0.5 0.5 1"/>
    <material name="brown" rgba="0.6 0.3 0.1 1"/>
    <material name="orange" rgba="0.8 0.5 0.2 1"/>
    <material name="pink" rgba="0.8 0.75 0.8 1"/>
    <material name="purple" rgba="0.5 0.2 0.5 1"/>
    <material name="lime" rgba="0.5 0.8 0.2 1"/>
    <material name="turquoise" rgba="0.25 0.88 0.82 1"/>
    <material name="gold" rgba="0.8 0.84 0.2 1"/>
    <material name="matplotlib_green" rgba="0.0 0.502 0.0 1"/>
    <material name="matplotlib_blue" rgba="0.012 0.263 0.8745 1"/>
    <material name="matplotlib_lightblue" rgba="0.482 0.784 0.9647 1"/>
    <material name="matplotlib_salmon" rgba="0.98 0.502 0.447 1"/>
  </asset>

  <visual>
    <headlight ambient=".4 .4 .4" diffuse=".8 .8 .8" specular="0.1 0.1 0.1"/>
    <map znear=".01"/>
    <quality shadowsize="2048"/>
    <global offwidth="1920" offheight="1080"/>
  </visual>

<worldbody>
<light pos="0 0 5" dir="0 0 -1"/>
<camera pos="3 0 1.5" mode="trackcom"/>
<geom name="floor" pos="0 0 -0.5" size="0 0 1" type="plane" material="matplane"/>
{inside_worldbody}
</worldbody>
</mujoco>
"""
    return mujoco.MjModel.from_xml_string(xml_str)


def _xml_str_one_body(body_number: int, geoms: list[Geometry]) -> str:
    inside_body = ""
    for geom in geoms:
        inside_body += _xml_str_one_geom(geom)
    return f"""
<body name="{body_number}" mocap="true">
{inside_body}
</body>
"""


def _xml_str_one_geom(geom: Geometry) -> str:
    if isinstance(geom.color, tuple):
        if len(geom.color) == 3:
            color = geom.color + (1.0,)
        else:
            color = geom.color
        rgba_material = f'rgba="{_array_to_str(color)}"'
    else:
        rgba_material = f'material="{geom.color}"'

    if isinstance(geom, Box):
        type_size = f'type="box" size="{_array_to_str([geom.dim_x / 2, geom.dim_y / 2, geom.dim_z / 2])}"'  # noqa: E501
    elif isinstance(geom, Sphere):
        type_size = f'type="sphere" size="{_array_to_str([geom.radius])}"'
    elif isinstance(geom, Capsule):
        type_size = (
            f'type="capsule" size="{_array_to_str([geom.radius, geom.length / 2])}"'
        )
    elif isinstance(geom, Cylinder):
        type_size = (
            f'type="cylinder" size="{_array_to_str([geom.radius, geom.length / 2])}"'
        )
    else:
        raise NotImplementedError

    return f"""
<geom {type_size} {rgba_material} pos="{_array_to_str(geom.transform.pos)}" quat="{_array_to_str(geom.transform.rot)}"/>
"""  # noqa: E501


def _array_to_str(arr: Sequence[float]) -> str:
    return "".join(["{:.2f} ".format(value) for value in arr])[:-1]


class MujocoScene:
    def init(self, geoms: list[Geometry]):
        self._body_names = list(set([geom.link_idx for geom in geoms]))
        self._model = _build_model_of_geoms(geoms)
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, 480, 640)

    def update(self, x: Transform):
        for body_name in self._body_names:
            # body name is just the str(parent_id)
            parent_id = int(body_name)
            mocap_id = int(self._model.body(body_name).mocapid)
            mocap_pos = x.pos[parent_id]
            mocap_quat = x.rot[parent_id]
            self._data.mocap_pos[mocap_id] = mocap_pos
            self._data.mocap_quat[mocap_id] = mocap_quat

    def render(self):
        mujoco.mj_forward(self._model, self._data)
        self._renderer.update_scene(self._data)
        return self._renderer.render()
