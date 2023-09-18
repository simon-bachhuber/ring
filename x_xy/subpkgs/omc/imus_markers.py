from functools import cache

import numpy as np

from .utils import resample


def _sync_imu_offset_with_optical(
    imu: dict, q_opt: np.ndarray, hz_imu: float, hz_opt: float
) -> int:
    from qmt import syncOptImu

    sync_info = syncOptImu(
        opt_quat=q_opt,
        opt_rate=hz_opt,
        imu_gyr=imu["gyr"],
        imu_rate=hz_imu,
        params=dict(syncRate=1000.0, cut=0.15, fc=10.0, correlate="rmse", fast=True),
    )
    imu_offset_time = sync_info["sync"][0][-1]
    return int(imu_offset_time * hz_imu)


def _imu_measurements_from_txt(
    path_imu,
    imu_file_prefix,
    imu_number: str,
    txt_file_delimiter: str = "\t",
    txt_file_skiprows: int = 4,
):
    from pathlib import Path

    import pandas as pd

    try:
        df = pd.read_csv(
            Path(path_imu).joinpath(
                Path(imu_file_prefix + imu_number).with_suffix(".txt")
            ),
            delimiter=txt_file_delimiter,
            skiprows=txt_file_skiprows,
        )
    except FileNotFoundError:
        return

    acc = df[["Acc_X", "Acc_Y", "Acc_Z"]].to_numpy()
    gyr = df[["Gyr_X", "Gyr_Y", "Gyr_Z"]].to_numpy()
    mag = df[["Mag_X", "Mag_Y", "Mag_Z"]].to_numpy()

    assert np.all(~np.isnan(acc))
    assert np.all(~np.isnan(gyr))
    assert np.all(~np.isnan(mag))

    return {"acc": acc, "gyr": gyr, "mag": mag}


@cache
def _load_df(path_optitrack: str):
    import pandas as pd

    print(f"Loading OMC data from file {path_optitrack}")
    _df_optitrack = pd.read_csv(path_optitrack, low_memory=False, skiprows=3)
    return _df_optitrack


def _get_marker_xyz(path_optitrack: str, seg_number: int, marker_number: int):
    df_optitrack = _load_df(path_optitrack)

    col = f"Segment_{seg_number}:Marker{marker_number}"
    x = df_optitrack[col].iloc[3:].to_numpy()
    y = df_optitrack[col + ".1"].iloc[3:].to_numpy()
    z = df_optitrack[col + ".2"].iloc[3:].to_numpy()

    return np.stack((x, y, z)).T.astype(np.float64)


def _construct_quat_from_three_markers(
    path_optitrack,
    seg_number: int,
    xaxis_marker_numbers: tuple[int],
    yaxis_marker_numbers: tuple[int],
    marker_imu_setup: dict,
):
    from qmt import quatFrom2Axes

    # >> Begin checks
    xaxis_marker_numbers, yaxis_marker_numbers = map(
        set, (xaxis_marker_numbers, yaxis_marker_numbers)
    )
    assert xaxis_marker_numbers != yaxis_marker_numbers
    assert len(xaxis_marker_numbers) == len(yaxis_marker_numbers) == 2
    xaxis_marker_numbers, yaxis_marker_numbers = map(
        list, (xaxis_marker_numbers, yaxis_marker_numbers)
    )

    get_relative_marker_pos = lambda nr, xy: marker_imu_setup[f"seg{seg_number}"][
        "position"
    ][nr - 1][xy]
    xs = set([get_relative_marker_pos(nr, 0) for nr in xaxis_marker_numbers])
    ys = set([get_relative_marker_pos(nr, 1) for nr in yaxis_marker_numbers])

    assert len(xs) > 1
    assert len(ys) > 1
    # << End checks

    delta_y_xaxis = get_relative_marker_pos(
        xaxis_marker_numbers[0], 1
    ) - get_relative_marker_pos(xaxis_marker_numbers[1], 1)
    delta_x_yaxis = get_relative_marker_pos(
        yaxis_marker_numbers[0], 0
    ) - get_relative_marker_pos(yaxis_marker_numbers[1], 0)

    def axis_with_positive_xy_comp(m1, m2, xy):
        s1 = get_relative_marker_pos(m1, xy)
        s2 = get_relative_marker_pos(m2, xy)
        sign = 1 if s2 < s1 else -1
        return sign * (
            _get_marker_xyz(path_optitrack, seg_number, m1)
            - _get_marker_xyz(path_optitrack, seg_number, m2)
        )

    axis_with_x_comp = axis_with_positive_xy_comp(
        xaxis_marker_numbers[0], xaxis_marker_numbers[1], 0
    )
    axis_with_y_comp = axis_with_positive_xy_comp(
        yaxis_marker_numbers[0], yaxis_marker_numbers[1], 1
    )

    if delta_y_xaxis == 0 and delta_x_yaxis == 0:
        return quatFrom2Axes(axis_with_x_comp, axis_with_y_comp)

    zaxis = np.cross(axis_with_x_comp, axis_with_y_comp)

    if delta_y_xaxis == 0:
        quats = quatFrom2Axes(x=axis_with_x_comp, z=zaxis)
    elif delta_x_yaxis == 0:
        quats = quatFrom2Axes(y=axis_with_y_comp, z=zaxis)
    else:
        raise Exception(
            f"For Segment {seg_number} you have chosen a marker combination which has "
            "neither a pure x-axis nor a pure y-axis component. Can't determine x and y"
            " unit vectors then."
        )

    # get ride of nan values
    return resample(quats, 1.0, 1.0)


def _construct_pos_from_single_marker(
    path_optitrack,
    seg_number: int,
    marker_imu_setup: dict,
):
    xyz = _get_marker_xyz(
        path_optitrack,
        seg_number,
        marker_imu_setup[f"seg{seg_number}"]["pos_single_marker"],
    )

    # get ride of nan values
    xyz = resample(xyz, 1.0, 1.0)

    # milimeters -> meters
    return xyz / 1000
