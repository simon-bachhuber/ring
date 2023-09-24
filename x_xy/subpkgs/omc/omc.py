"""
This module allows to read in Optical Motion Capture (OMC) and IMU data by
    - synchronizing both systems, and
    - constructing quaternions by spanning an orthongal coordinate system.
"""

import json
from typing import Optional
import warnings

import jax
import numpy as np
import qmt

from x_xy.utils import parse_path

from .imus_markers import _construct_pos_from_single_marker
from .imus_markers import _construct_quat_from_three_markers
from .imus_markers import _imu_measurements_from_txt
from .imus_markers import _sync_imu_offset_with_optical
from .utils import autodetermine_imu_file_delimiter
from .utils import autodetermine_imu_file_prefix
from .utils import autodetermine_imu_freq
from .utils import autodetermine_optitrack_freq


def read_omc(
    path_marker_imu_setup_file: str,
    path_optitrack_file: str,
    path_imu_folder: str,
    imu_file_prefix: Optional[str] = None,
    imu_file_delimiter: Optional[str] = None,
    # zyx convention
    qEOpt2EImu_euler_deg: np.ndarray = np.array([0.0, 0, 0]),
    # if imu and seg not in `q_Imu2seg[seg][imu]`, then [0, 0, 0]
    # also zyx convention
    qImu2Seg_euler_deg: dict = {},
    imu_sync_offset: Optional[int] = None,
    hz_opt: Optional[int] = None,
    hz_imu: Optional[int] = None,
    verbose: bool = True,
    assume_imus_synced: bool = False,
) -> dict:
    p_setup_file = parse_path(path_marker_imu_setup_file, extension="json")
    path_optitrack = parse_path(path_optitrack_file, extension="csv")
    path_imu = parse_path(path_imu_folder)

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

    if imu_file_prefix is None:
        imu_file_prefix = autodetermine_imu_file_prefix(path_imu)
        if verbose:
            print(f"IMU File Prefix: {imu_file_prefix}")

    if imu_file_delimiter is None:
        imu_file_delimiter = autodetermine_imu_file_delimiter(path_imu)
        if verbose:
            print(f"IMU File Delimiter: {imu_file_delimiter}")

    if imu_sync_offset is not None:
        if assume_imus_synced is False:
            assume_imus_synced = True
            warnings.warn("`assume_imus_synced` was overwritten to `True`.")

    data = {}
    for seg in marker_imu_setup["segments"]:
        data[seg] = {}
        seg_number = int(seg[3])
        xaxis_markers = marker_imu_setup[seg]["xaxis_markers"][0]
        yaxis_markers = marker_imu_setup[seg]["yaxis_markers"][0]

        quat_opt_markers2EOpt = _construct_quat_from_three_markers(
            path_optitrack, seg_number, xaxis_markers, yaxis_markers, marker_imu_setup
        )

        imus = {}
        for imu in marker_imu_setup["imus"]:
            imu_number = marker_imu_setup[seg][imu]
            imu_unsynced = _imu_measurements_from_txt(
                path_imu, imu_file_prefix, imu_number, imu_file_delimiter
            )
            if imu_sync_offset is None:
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

            # alignment: rigid-imu to markers
            q_Imu2Seg_default = np.array([1.0, 0, 0, 0])
            if seg in qImu2Seg_euler_deg:
                if imu in qImu2Seg_euler_deg[seg]:
                    q_Imu2Seg_default = _from_euler(qImu2Seg_euler_deg[seg][imu])

            for signal in ["acc", "mag", "gyr"]:
                imus[imu][signal] = qmt.rotate(q_Imu2Seg_default, imus[imu][signal])

            # reset `imu_sync_offset` is required
            if not assume_imus_synced:
                imu_sync_offset = None

        data[seg].update(imus)

        # alignment: earth_omc to earth_inertial
        qEOpt2EImu = _from_euler(qEOpt2EImu_euler_deg)
        data[seg]["quat"] = qmt.qmult(qEOpt2EImu, quat_opt_markers2EOpt)
        for marker_number in range(1, 5):
            data[seg][f"marker{marker_number}"] = qmt.rotate(
                qEOpt2EImu,
                _construct_pos_from_single_marker(
                    path_optitrack, seg_number, marker_number
                ),
            )

    return data


def _from_euler(angles_deg: np.ndarray):
    return qmt.quatFromEulerAngles(np.deg2rad(angles_deg))
