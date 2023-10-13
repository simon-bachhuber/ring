from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import joblib
import tree_utils
import yaml

import x_xy
from x_xy import maths
from x_xy.io import load_comments_from_str
from x_xy.io import load_comments_from_xml
from x_xy.subpkgs import omc
from x_xy.subpkgs import sys_composer

_id2xml = {"S_06": "setups/arm.xml"}


def _relative_to_this_file(path: str) -> Path:
    return Path(__file__).parent.joinpath(path)


def _read_yaml(path: str):
    with open(_relative_to_this_file(path)) as file:
        yaml_str = yaml.safe_load(file)
    return yaml_str


def _replace_rxyz_with(sys: x_xy.base.System, replace_with: str):
    return sys.replace(
        link_types=[
            replace_with if (typ in ["rx", "ry", "rz"]) else typ
            for typ in sys.link_types
        ]
    )


def _morph_new_parents_from_xml_file(file_path: str) -> dict[str, list]:
    comments = load_comments_from_xml(file_path, key="morph")
    seg_new_parents_map = {}
    for comment in comments:
        segi, new_parents = comment["key"], comment["parents"]
        seg_new_parents_map[segi] = eval(new_parents)
    return seg_new_parents_map


def load_sys(
    exp_id: str,
    preprocess_sys: Optional[Callable] = None,
    morph_yaml_key: Optional[str] = None,
    delete_after_morph: Optional[list[str]] = None,
    replace_rxyz: Optional[str] = None,
) -> x_xy.base.System:
    xml_path = _relative_to_this_file(_id2xml[exp_id])
    sys = x_xy.io.load_sys_from_xml(xml_path)

    if preprocess_sys is not None:
        sys = preprocess_sys(sys)

    if replace_rxyz is not None:
        sys = _replace_rxyz_with(sys, replace_rxyz)

    if morph_yaml_key is not None:
        new_parents = _morph_new_parents_from_xml_file(xml_path)[morph_yaml_key]
        sys = sys_composer.morph_system(sys, new_parents)

    if delete_after_morph is not None:
        sys = sys_composer.delete_subsystem(sys, delete_after_morph)

    return sys


def load_data(
    exp_id: str,
    motion_start: str,
    motion_stop: Optional[str] = None,
    left_padd: float = 0.0,
    right_padd: float = 0.0,
    start_for_start: bool = True,
    stop_for_stop: bool = True,
    resample_to_hz: float = 100.0,
) -> dict:
    trial_data = joblib.load(
        _relative_to_this_file(f"joblib/synced_aligned_nan_croptail_{exp_id}.joblib")
    )

    metadata = _read_yaml("metadata.yaml")[exp_id]
    timings = metadata["timings"]
    hz_imu, hz_omc = float(metadata["hz"]["imu"]), float(metadata["hz"]["omc"])

    trial_data = omc.resample(
        trial_data,
        hz_in=omc.hz_helper(trial_data.keys(), hz_imu=hz_imu, hz_omc=hz_omc),
        hz_out=resample_to_hz,
    )

    if motion_stop is None:
        motion_stop = motion_start

    motions = list(timings.keys())
    assert motions.index(motion_start) <= motions.index(
        motion_stop
    ), f"starting point motion {motion_start} is after the stopping "
    "point motion {motion_stop}"

    if motion_start == motion_stop:
        assert start_for_start and stop_for_stop, "Empty sequence, stop <= start"

    t1 = timings[motion_start]["start" if start_for_start else "stop"] - left_padd
    # ensure that t1 >= 0
    t1 = max(t1, 0.0)
    t2 = timings[motion_stop]["stop" if stop_for_stop else "start"] + right_padd

    trial_data = _crop_sequence(trial_data, 1 / resample_to_hz, t1=t1, t2=t2)

    return trial_data


def link_name_pos_rot_data(data: dict, xml_str: str) -> dict:
    comments = load_comments_from_str(xml_str, key="omc")

    data_out = dict()
    for comment in comments:
        bodyname, omcname, marker = (
            comment["bodyname"],
            comment["omcname"],
            int(comment["marker"]),
        )

        if "pos" not in comment:
            pos_offset = jnp.zeros((3,))
        else:
            pos_x, pos_y, pos_z = map(float, comment["pos"].split(","))
            pos_offset = jnp.array([pos_x, pos_y, pos_z])
        pos = data[omcname][f"marker{marker}"]
        quat = data[omcname]["quat"]
        pos_with_offset = pos + maths.rotate(pos_offset, quat)
        data_out[bodyname] = dict(pos=pos_with_offset, quat=quat)

    return data_out


def _crop_sequence(data: dict, dt: float, t1: float = 0.0, t2: Optional[float] = None):
    # crop time left and right
    if t2 is None:
        t2i = tree_utils.tree_shape(data)
    else:
        t2i = int(t2 / dt)
    t1i = int(t1 / dt)
    return jax.tree_map(lambda arr: jnp.array(arr)[t1i:t2i], data)
