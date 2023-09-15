"""
This module allows to preprocess Optical Motion Capture (OMC) data by
    - re-sampling IMUs and Optitrack to a common frequency
    - synchronizing both systems
    - filling in NaN values
    - constructing quaternions by spanning an orthongal coordinate system
"""

import json
import os
from pathlib import Path
from typing import Optional

import jax
import joblib
import numpy as np
from scipy.io import savemat
import tree_utils

from x_xy.utils import parse_path

from .imus import _imu_measurements_from_txt
from .markers import _construct_pos_from_single_marker
from .markers import _construct_quat_from_four_markers


def dump_omc(
    path_marker_imu_setup_file: str,
    path_optitrack_file: str,
    path_imu_folder: str,
    path_output: str,
    imu_file_prefix: str = "MT_012102D5-000-000_00B483",
    imu_file_delimiter: str = "\t",
    hz_optitrack: Optional[int] = None,
    hz_imu: Optional[int] = None,
    hz_common: int = 100,
    verbose: bool = True,
    assume_imus_are_in_sync: bool = False,
    save_as_matlab: bool = False,
):
    try:
        import qmt
    except ImportError:
        print(
            "This module requires the `qmt` library to be installed, "
            "use `pip install qmt`"
        )

    p_setup_file = parse_path(path_marker_imu_setup_file, extension="json")
    path_optitrack = parse_path(path_optitrack_file, extension="csv")
    path_imu = parse_path(path_imu_folder)
    p_output = parse_path(path_output, extension="joblib")

    with open(p_setup_file) as f:
        marker_imu_setup = json.load(f)

    data = {}

    imu_offset_time = None
    for seg in marker_imu_setup:
        # special case; This json field just stores that one quaternion
        if seg == "earth_omc_to_earth_inertial":
            continue

        opt_imu_seg_data, imu_offset_time = _synced_opti_imu_data(
            path_optitrack,
            path_imu,
            imu_file_prefix,
            int(seg[-1]),
            marker_imu_setup,
            hz_imu if hz_imu else autodetermine_imu_freq(path_imu),
            hz_optitrack
            if hz_optitrack
            else autodetermine_optitrack_freq(path_optitrack),
            hz_common,
            imu_file_delimiter=imu_file_delimiter,
            verbose=verbose,
            imu_offset_time=imu_offset_time,
            assume_imus_are_in_sync=assume_imus_are_in_sync,
        )

        if opt_imu_seg_data is None:
            print(f"Segment_{seg[-1]} was not found in OMC data.")
            continue

        opt_imu_seg_data_aligned = dict()
        opt_imu_seg_data_aligned["imu_rigid"] = dict()

        pos_EOpt, quat_markers2EOpt = opt_imu_seg_data["pos"], opt_imu_seg_data["quat"]

        # alignment: earth_omc to earth_inertial
        q_EOpt2EInert = _str_to_numpy_array(
            marker_imu_setup["earth_omc_to_earth_inertial"]
        )
        opt_imu_seg_data_aligned["quat"] = qmt.qinv(
            qmt.qmult(q_EOpt2EInert, quat_markers2EOpt)
        )
        opt_imu_seg_data_aligned["pos"] = qmt.rotate(q_EOpt2EInert, pos_EOpt)

        # alignment: rigid-imu to markers
        q_imu2markers = _str_to_numpy_array(
            marker_imu_setup[seg]["imu_rigid_imu_to_markers"]
        )
        for signal in ["acc", "mag", "gyr"]:
            opt_imu_seg_data_aligned["imu_rigid"][signal] = qmt.rotate(
                q_imu2markers, opt_imu_seg_data["imu_rigid"][signal]
            )

        opt_imu_seg_data_aligned["imu_flex"] = opt_imu_seg_data["imu_flex"]

        data[seg] = opt_imu_seg_data_aligned

    joblib.dump(data, p_output)
    if save_as_matlab:
        savemat(parse_path(p_output, extension="mat"), data)


def autodetermine_imu_freq(path_imu) -> int:
    hz = []
    for file in os.listdir(path_imu):
        file = Path(path_imu).joinpath(file)
        if file.suffix != ".txt":
            continue

        with open(file) as f:
            f.readline()
            # second line in txt file is: // Update Rate: 40.0Hz
            second_line = f.readline()
            before = len("// Update Rate:")
            hz.append(int(float(second_line[before:-3])))

    assert len(set(hz)) == 1, f"IMUs have multiple sampling rates {hz}"
    return hz[0]


def autodetermine_optitrack_freq(path_optitrack):
    def find_framerate_in_line(line: str, key: str):
        before = line.find(key) + len(key) + 1
        return int(float(line[before:].split(",")[0]))

    # first line is:
    # ...,Capture Frame Rate,120.000000,Export Frame Rate,120.000000,...
    with open(path_optitrack) as f:
        line = f.readline()
        hz_cap = find_framerate_in_line(line, "Capture Frame Rate")
        hz_exp = find_framerate_in_line(line, "Export Frame Rate")
        assert hz_cap == hz_exp, "Capture and exported frame rate are not equal"

    return hz_exp


def _find_imu_keys(seg_number, marker_imu_setup) -> list[str]:
    cond = lambda key: (key[:3] == "imu" and key != "imu_rigid_imu_to_markers")
    return [key for key in marker_imu_setup[f"seg{seg_number}"] if cond(key)]


def _tree_slice(tree, start=None, stop=None):
    return jax.tree_map(lambda arr: arr[start:stop], tree)


def _synced_opti_imu_data(
    path_optitrack,
    path_imu,
    imu_file_prefix,
    seg_number: int,
    marker_imu_setup: dict,
    hz_imus,
    hz_opti,
    hz_common,
    imu_file_delimiter,
    verbose: bool = True,
    imu_offset_time: Optional[float] = None,
    assume_imus_are_in_sync: bool = False,
):
    from qmt import syncOptImu

    q = _construct_quat_from_four_markers(
        path_optitrack,
        seg_number,
        marker_imu_setup,
        hz_opti,
        hz_common,
        verbose,
        resample=True,
    )

    # then `seg_number` is not present in .csv of OMC data
    if q is None:
        return None, imu_offset_time

    pos = _construct_pos_from_single_marker(
        path_optitrack, seg_number, marker_imu_setup, hz_opti, hz_common, resample=True
    )

    imus = {}
    N_imu = 1e16
    for imu_key in _find_imu_keys(seg_number, marker_imu_setup):
        imu_data = _imu_measurements_from_txt(
            path_imu,
            imu_file_prefix,
            marker_imu_setup[f"seg{seg_number}"][imu_key],
            hz_imus,
            hz_common,
            resample=True,
            txt_file_delimiter=imu_file_delimiter,
        )

        # then this IMU was not recording in this trial
        if imu_data is None:
            print(f"For segment{seg_number} IMU {imu_key} was not recorded.")
            continue

        if verbose:
            print(">>> SYNC START (might take 1-2 minutes)")

        if not assume_imus_are_in_sync:
            # delete previous sync, and re-calculate for each IMU
            imu_offset_time = None

        if imu_offset_time is None:
            sync_info = syncOptImu(
                opt_quat=q,
                opt_rate=hz_common,
                imu_gyr=imu_data["gyr"],
                imu_rate=hz_common,
                params=dict(
                    syncRate=1000.0, cut=0.15, fc=10.0, correlate="rmse", fast=True
                ),
            )
            imu_offset_time = sync_info["sync"][0][-1]
        else:
            sync_info = {}

        if verbose:
            print("<<< SYNC FINISH")
        assert (
            imu_offset_time > 0
        ), "Time shift between Optitrack and IMUs is negative. Why?"
        imu_offset_index = int(imu_offset_time * hz_common)

        N_this_imu = tree_utils.tree_shape(imu_data, 0)
        imus[imu_key] = _tree_slice(imu_data, imu_offset_index, N_this_imu)
        N_this_imu -= imu_offset_index
        N_imu = min(N_imu, N_this_imu)

        if verbose:
            print("---AFTER SYNC---")
            print("SYNC INFO: ", sync_info)
            print(f"IMU offset has been determined to be t={imu_offset_time}")
            print(
                "IMU sequence has been shifted by an "
                f"index offset of {imu_offset_index}"
            )
            print(f"After offset shift IMU sequences has {N_this_imu} values")

    N_q = q.shape[0]
    N = min(N_imu, N_q)
    q, imus = _tree_slice((q, imus), 0, N)

    if verbose:
        print(f"Quaternion sequence has {N_q} values")
        print(f"Crop sequence at the end to reach a common length of {N} values")

    return {"quat": q, "pos": pos, **imus}, imu_offset_time


def _str_to_numpy_array(expr: str) -> np.ndarray:
    return np.array([float(num) for num in expr.split(" ")], dtype=float)
