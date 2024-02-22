from functools import cache
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
from x_xy.io.xml.from_xml import _load_xml
from x_xy.subpkgs import sys_composer

# TODO exp imports omc; pip install x_xy[exp] works still correctly because of setup.py
from x_xy.subpkgs.omc import utils as omc_utils

arm_xml = "setups/arm.xml"
gait_xml = "setups/gait.xml"
_id2xml = {
    "S_04": arm_xml,
    "S_06": arm_xml,
    "S_07": arm_xml,
    "S_08": arm_xml,
    "S_09": arm_xml,
    "S_10": arm_xml,
    "S_12": gait_xml,
    "S_13": gait_xml,
    "S_14": gait_xml,
    "S_15": gait_xml,
    "S_16": gait_xml,
    "T_01": gait_xml,
}


def _relative_to_this_file(path: str) -> Path:
    return Path(__file__).parent.joinpath(path)


def _read_yaml(path: str):
    with open(_relative_to_this_file(path)) as file:
        yaml_str = yaml.safe_load(file)
    return yaml_str


def _replace_rxyz_with(sys: x_xy.base.System, replace_with: str):
    if replace_with == "rr":
        new_damp = jnp.array([3.0])
    elif replace_with == "rr_imp":
        new_damp = jnp.array([3.0, 3.0])
    else:
        raise Exception()

    for name, typ in zip(sys.link_names, sys.link_types):
        if typ in ["rx", "ry", "rz"]:
            sys = sys.change_joint_type(name, replace_with, new_damp=new_damp)

    return sys


def _morph_new_parents_from_xml_file(file_path: str) -> dict[str, list]:
    comments = load_comments_from_xml(file_path, key="morph")
    seg_new_parents_map = {}
    for comment in comments:
        segi, new_parents = comment["key"], comment["parents"]
        seg_new_parents_map[segi] = eval(new_parents)
    return seg_new_parents_map


def load_arm_or_gait(exp_id: str) -> str:
    "Returns either `arm` or `gait`"
    xml = _id2xml[exp_id]
    if xml == arm_xml:
        return "arm"
    return "gait"


def load_xml_str(exp_id: str) -> str:
    return _load_xml(_relative_to_this_file(_id2xml[exp_id]))


@cache
def load_sys(
    exp_id: str,
    preprocess_sys: Optional[Callable] = None,
    morph_yaml_key: Optional[str] = None,
    delete_after_morph: Optional[tuple[str]] = None,
    replace_rxyz: Optional[str] = None,
) -> x_xy.base.System:
    xml_path = _relative_to_this_file(_id2xml[exp_id])
    sys = x_xy.io.load_sys_from_xml(xml_path)

    if preprocess_sys is not None:
        sys = preprocess_sys(sys)

    if replace_rxyz is not None:
        sys = _replace_rxyz_with(sys, replace_rxyz)

    # save to xml_str and reload such that joint_params get populated
    sys = x_xy.load_sys_from_str(x_xy.save_sys_to_str(sys))

    if morph_yaml_key is not None:
        skip_morph = False
        if load_arm_or_gait(exp_id) == "arm" and morph_yaml_key == "seg1":
            skip_morph = True
        if load_arm_or_gait(exp_id) == "gait" and morph_yaml_key == "seg4":
            skip_morph = True

        if not skip_morph:
            new_parents = _morph_new_parents_from_xml_file(xml_path)[morph_yaml_key]
            sys = sys_composer.morph_system(sys, new_parents)

    if delete_after_morph is not None:
        sys = sys_composer.delete_subsystem(sys, list(delete_after_morph))

    return sys


@cache
def load_data(
    exp_id: str,
    motion_start: Optional[str] = None,
    motion_stop: Optional[str] = None,
    left_padd: float = 0.0,
    right_padd: float = 0.0,
    resample_to_hz: float = 100.0,
) -> dict:
    trial_data = joblib.load(x_xy.utils.download_from_repo(f"data/{exp_id}.joblib"))

    metadata = _read_yaml("metadata.yaml")[exp_id]
    timings = metadata["timings"]
    hz_imu, hz_omc = float(metadata["hz"]["imu"]), float(metadata["hz"]["omc"])

    trial_data = omc_utils.resample(
        trial_data,
        hz_in=omc_utils.hz_helper(trial_data.keys(), hz_imu=hz_imu, hz_omc=hz_omc),
        hz_out=resample_to_hz,
        vecinterp_method="cubic",
    )
    trial_data = omc_utils.crop_tail(
        trial_data, resample_to_hz, strict=True, verbose=False
    )

    if motion_start is not None:
        assert (
            motion_start in timings
        ), f"`{motion_start}` is not one of {load_timings(exp_id).keys()}"

        motion_sequence = list(timings.keys())
        next_motion_i = motion_sequence.index(motion_start) + 1
        assert next_motion_i < len(motion_sequence)

        if motion_stop is None:
            motion_stop = motion_sequence[next_motion_i]

        assert (
            motion_stop in timings
        ), f"`{motion_stop}` is not one of {load_timings(exp_id).keys()}"

        assert motion_sequence.index(motion_start) < motion_sequence.index(
            motion_stop
        ), "Empty sequence, stop <= start"

        t1 = timings[motion_start] - left_padd
        # ensure that t1 >= 0
        t1 = max(t1, 0.0)
        t2 = timings[motion_stop] + right_padd

        trial_data = _crop_sequence(trial_data, 1 / resample_to_hz, t1=t1, t2=t2)
    else:
        assert motion_stop is None

    return trial_data


def load_timings(exp_id: str) -> dict[str, float]:
    return _read_yaml("metadata.yaml")[exp_id]["timings"]


def load_hz_omc(exp_id: str) -> int:
    return int(_read_yaml("metadata.yaml")[exp_id]["hz"]["omc"])


def load_hz_imu(exp_id: str) -> int:
    return int(_read_yaml("metadata.yaml")[exp_id]["hz"]["imu"])


def link_name_pos_rot_data(data: dict, xml_str: str) -> dict:
    comments = load_comments_from_str(xml_str, key="omc")

    data_out = dict()
    for comment in comments:
        bodyname, omcname, marker = (
            comment["bodyname"],
            comment["omcname"],
            int(comment["marker"]),
        )

        # imagine we want to use the `arm.xml` for the trial S_04
        # it does not have data of 5 segments but only of 3
        # but that's okay since `data_out` is then usually feed into the function
        # `xs_from_raw` which then gets the actual system at hand
        if omcname not in data:
            continue

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
