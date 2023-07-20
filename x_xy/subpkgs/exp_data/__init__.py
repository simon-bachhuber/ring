import os
from pathlib import Path
from typing import Optional

import joblib
import yaml

import x_xy
from x_xy.subpkgs import sim2real, sys_composer
from x_xy.utils import parse_path

from .omc_to_joblib import HZ, exp_dir


def _load_file_path(exp_id: str, extension: str):
    return exp_dir(parse_path(exp_id, extension="", mkdir=False)).joinpath(
        parse_path(exp_id, mkdir=False, extension=extension)
    )


def _read_yaml(exp_id):
    with open(_load_file_path(exp_id, "yaml")) as file:
        yaml_str = yaml.safe_load(file)
    return yaml_str


def load_sys(
    exp_id: str,
    morph_yaml_key: Optional[str] = None,
    delete_after_morph: Optional[list[str]] = None,
) -> x_xy.base.System:
    xml_path = _load_file_path(exp_id, "xml")
    sys = x_xy.io.load_sys_from_xml(xml_path)

    if morph_yaml_key is not None:
        new_parents = _read_yaml(exp_id)["morph"][morph_yaml_key]
        sys = sys_composer.morph_system(sys, new_parents)

    if delete_after_morph is not None:
        sys = sys_composer.delete_subsystem(sys, delete_after_morph)

    return sys


def list_experiments() -> list[str]:
    exps = []
    parent = Path(__file__).parent
    for child in os.listdir(parent):
        file = parent.joinpath(child)
        if file.is_dir():
            if child[:2] != "__":
                exps.append(child)
    exps.sort()
    return exps


def load_data(
    exp_id: str,
    motion_start: str,
    motion_stop: Optional[str] = None,
    left_padd: float = 0.0,
    right_padd: float = 0.0,
    start_for_start: bool = True,
    stop_for_stop: bool = True,
) -> dict:
    trial_data = joblib.load(_load_file_path(exp_id, "joblib"))

    timings = _read_yaml(exp_id)["timings"]

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

    return sim2real._crop_sequence(trial_data, 1 / HZ, t1=t1, t2=t2)
