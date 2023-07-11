from typing import Optional

import joblib
import yaml

import x_xy
from x_xy.subpkgs import sim2real
from x_xy.utils import parse_path

from .omc_to_joblib import HZ, exp_dir, omc_to_joblib


def _load_file_path(exp_id: str, extension: str):
    return exp_dir(parse_path(exp_id, extension="", mkdir=False)).joinpath(
        parse_path(exp_id, mkdir=False, extension=extension)
    )


def load_sys(exp_id: str) -> x_xy.base.System:
    xml_path = _load_file_path(exp_id, "xml")
    return x_xy.io.load_sys_from_xml(xml_path)


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

    with open(_load_file_path(exp_id, "yaml")) as file:
        timings = yaml.safe_load(file)

    if motion_stop is None:
        motion_stop = motion_start

    motions = list(timings.keys())
    assert motions.index(motion_start) <= motions.index(
        motion_stop
    ), f"starting point motion {motion_start} is after the stopping "
    "point motion {motion_stop}"

    if motion_start == motion_stop:
        assert start_for_start and stop_for_stop, "Empty sequence, stop <= start"

    t1 = timings[motion_start]["start" if start_for_start else "stop"] + left_padd
    # ensure that t1 >= 0
    t1 = max(t1, 0.0)
    t2 = timings[motion_stop]["stop" if stop_for_stop else "start"] + right_padd

    return sim2real._crop_sequence(trial_data, 1 / HZ, t1=t1, t2=t2)
