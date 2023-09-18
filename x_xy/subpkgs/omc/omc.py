"""
This module allows to preprocess Optical Motion Capture (OMC) data by
    - synchronizing both systems
    - filling in NaN values
    - constructing quaternions by spanning an orthongal coordinate system
"""

import json
from typing import Optional

import jax
import joblib
import numpy as np
from scipy.io import savemat

from x_xy.utils import parse_path

from .imus_markers import _construct_pos_from_single_marker
from .imus_markers import _construct_quat_from_three_markers
from .imus_markers import _imu_measurements_from_txt
from .imus_markers import _sync_imu_offset_with_optical
from .utils import autodetermine_imu_freq
from .utils import autodetermine_optitrack_freq


def dump_omc(
    path_marker_imu_setup_file: str,
    path_optitrack_file: str,
    path_imu_folder: str,
    path_output: str,
    imu_file_prefix: str = "MT_012102D5-000-000_00B483",
    imu_file_delimiter: str = "\t",
    hz_opt: Optional[int] = None,
    hz_imu: Optional[int] = None,
    verbose: bool = True,
    save_as_matlab: bool = False,
):
    try:
        import qmt
    except ImportError:
        print(
            "This module requires the `qmt` library to be installed, "
            "use `pip install qmt`"
        )
    try:
        import pandas

        del pandas
    except ImportError:
        print(
            "This module requires the `pandas` library to be installed, "
            "use `pip install pandas`"
        )

    p_setup_file = parse_path(path_marker_imu_setup_file, extension="json")
    path_optitrack = parse_path(path_optitrack_file, extension="csv")
    path_imu = parse_path(path_imu_folder)
    p_output = parse_path(path_output, extension="joblib")

    with open(p_setup_file) as f:
        marker_imu_setup = json.load(f)

    if hz_opt is None:
        hz_opt = autodetermine_optitrack_freq(path_optitrack)
        if verbose:
            print(f"OMC Hz: {hz_opt}")

    if hz_imu is None:
        hz_imu = autodetermine_imu_freq(path_imu)
        if verbose:
            print(f"IMU Hz: {hz_imu}")

    data = {}
    for seg in marker_imu_setup["segments"]:
        data[seg] = {}
        seg_number = int(seg[3])
        xaxis_markers = marker_imu_setup[seg]["xaxis_markers"][0]
        yaxis_markers = marker_imu_setup[seg]["yaxis_markers"][0]

        quat_opt_markers2EOpt = _construct_quat_from_three_markers(
            path_optitrack, seg_number, xaxis_markers, yaxis_markers, marker_imu_setup
        )
        pos_opt = _construct_pos_from_single_marker(
            path_optitrack, seg_number, marker_imu_setup
        )

        imus = {}
        for imu in marker_imu_setup["imus"]:
            imu_number = marker_imu_setup[seg][imu]
            imu_unsynced = _imu_measurements_from_txt(
                path_imu, imu_file_prefix, imu_number, imu_file_delimiter
            )
            imu_sync_offset = _sync_imu_offset_with_optical(
                imu_unsynced, quat_opt_markers2EOpt, hz_imu, hz_opt
            )
            if verbose:
                print(
                    f"Segment: {seg_number}, IMU: {imu_number}, Offset: "
                    f"{imu_sync_offset}"
                )

            assert imu_sync_offset >= 0, f"IMU sync offset negative, {imu_sync_offset}"
            imu_synced = jax.tree_map(lambda arr: arr[imu_sync_offset:], imu_unsynced)
            imus[imu] = imu_synced

            if f"{imu}_imu_to_markers" in marker_imu_setup[seg]:
                # alignment: rigid-imu to markers
                q_imu2markers = _str_to_numpy_array(
                    marker_imu_setup[seg][f"{imu}_imu_to_markers"]
                )
                for signal in ["acc", "mag", "gyr"]:
                    imus[imu][signal] = qmt.rotate(q_imu2markers, imus[imu][signal])

        data[seg].update(imus)

        # alignment: earth_omc to earth_inertial
        q_EOpt2EInert = _str_to_numpy_array(
            marker_imu_setup["earth_omc_to_earth_inertial"]
        )
        data[seg]["quat"] = qmt.qinv(qmt.qmult(q_EOpt2EInert, quat_opt_markers2EOpt))
        data[seg]["pos"] = qmt.rotate(q_EOpt2EInert, pos_opt)

    if verbose:
        print(f"Saving file {p_output}.")
    joblib.dump(data, p_output)

    if save_as_matlab:
        p_output = parse_path(p_output, extension="mat")
        if verbose:
            print(f"Saving file {p_output}.")
        savemat(p_output, data)


def _str_to_numpy_array(expr: str) -> np.ndarray:
    return np.array([float(num) for num in expr.split(" ")], dtype=float)
