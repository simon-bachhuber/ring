"""
This module allows to preprocess Optical Motion Capture (OMC) data by
    - re-sampling IMUs and Optitrack to a common frequency
    - synchronizing both systems
    - filling in NaN values
    - constructing quaternions by spanning an orthongal coordinate system
"""

import json
from typing import Optional

import jax
import joblib
import tree_utils
from scipy.io import savemat

from x_xy.io.omc.imus import _imu_measurements_from_txt
from x_xy.io.omc.markers import (
    _construct_pos_from_single_marker,
    _construct_quat_from_four_markers,
)
from x_xy.utils import parse_path


def process_omc(
    experiment_name: str,
    path_marker_imu_setup_file: str,
    path_optitrack_file: str,
    path_imu_folder: str,
    path_output_folder: str,
    imu_file_prefix: str = "MT_012102D5-000-000_00B483",
    hz_optitrack: int = 120,
    hz_imu: int = 40,
    hz_common: int = 100,
    verbose: bool = True,
):
    try:
        import qmt

        del qmt
    except ImportError:
        print(
            "This module requires the `qmt` library to be installed, "
            "use `pip install qmt`"
        )

    p_setup_file = parse_path(path_marker_imu_setup_file, "json")
    path_optitrack = parse_path(path_optitrack_file, "csv")
    path_imu = parse_path(path_imu_folder)
    p_output = parse_path(path_output_folder)

    with open(p_setup_file) as f:
        marker_imu_setup = json.load(f)

    data = {}

    for seg in marker_imu_setup:
        opt_imu_seg_data = _synced_opti_imu_data(
            path_optitrack,
            path_imu,
            imu_file_prefix,
            int(seg[-1]),
            marker_imu_setup,
            hz_imu,
            hz_optitrack,
            hz_common,
            verbose,
            imu_offset_time=3.33,
        )[0]

        if opt_imu_seg_data is not None:
            data[seg] = opt_imu_seg_data

    # make folder
    path_mat_file = parse_path(p_output + f"/{experiment_name}.mat")
    savemat(path_mat_file, data)
    joblib.dump(data, p_output + f"/{experiment_name}.joblib")


def _find_imu_keys(seg_number, marker_imu_setup) -> list[str]:
    cond = lambda key: key[:3] == "imu"
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
    verbose: bool = True,
    imu_offset_time: Optional[float] = None,
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
        return

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
        )

        # then this IMU was not recording in this trial
        if imu_data is None:
            continue

        if verbose:
            print(">>> SYNC START (might take 1-2 minutes)")

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
