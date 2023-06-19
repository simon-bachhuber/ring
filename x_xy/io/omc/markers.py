import numpy as np
import pandas as pd

from x_xy.io.omc.nan import _interp_nan_values, _nan_check, _slerp_nan_values

_df_optitrack = None
_path_optitrack = None


def _get_marker_xyz(path_optitrack: str, seg_number: int, marker_number: int):
    global _df_optitrack, _path_optitrack
    if _path_optitrack is not None:
        if path_optitrack != _path_optitrack:
            _path_optitrack = path_optitrack
            _df_optitrack = None

    if _df_optitrack is None:
        _df_optitrack = pd.read_csv(path_optitrack, low_memory=False, skiprows=3)

    col = f"Segment_{seg_number}:Marker{marker_number}"
    x = _df_optitrack[col].iloc[3:].to_numpy()
    y = _df_optitrack[col + ".1"].iloc[3:].to_numpy()
    z = _df_optitrack[col + ".2"].iloc[3:].to_numpy()

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
        return quatFrom2Axes(x=axis_with_x_comp, z=zaxis)
    elif delta_x_yaxis == 0:
        return quatFrom2Axes(y=axis_with_y_comp, z=zaxis)
    else:
        raise Exception(
            f"For Segment {seg_number} you have chosen a marker combination which has "
            "neither a pure x-axis nor a pure y-axis component. Can't determine x and y"
            " unit vectors then."
        )


def _construct_quat_from_four_markers(
    path_optitrack,
    seg_number: int,
    marker_imu_setup: dict,
    hz1,
    hz2,
    verbose: bool = True,
    resample: bool = True,
):
    from qmt import quatInterp

    setup = marker_imu_setup[f"seg{seg_number}"]
    q_estimate = []
    for xaxis_marker_nrs, yaxis_marker_nrs in zip(
        setup["xaxis_markers"], setup["yaxis_markers"]
    ):
        q_estimate.append(
            _construct_quat_from_three_markers(
                path_optitrack,
                seg_number,
                xaxis_marker_nrs,
                yaxis_marker_nrs,
                marker_imu_setup,
            )
        )

    q = q_estimate[0]
    for i, q_alt in enumerate(q_estimate[1:]):
        mask = np.isnan(q) * ~np.isnan(q_alt)
        assert _all_equal_in_last_axis(np.isnan(q))
        if verbose:
            print(
                f"For seg{seg_number} marker alternative {i+1} has provided"
                f" {np.sum(mask)} non-NaN values"
            )
        q[mask] = q_alt[mask]

    if verbose:
        _nan_check(q, seg_number, hz1)

    if resample:
        ind = np.arange(q.shape[0], step=hz1 / hz2)
        q = quatInterp(q, ind)
        q = _slerp_nan_values(q)
        if verbose:
            print("--- AFTER RESAMPLE ---")
            _nan_check(q, seg_number, hz2)

    return q


def _construct_pos_from_single_marker(
    path_optitrack,
    seg_number: int,
    marker_imu_setup: dict,
    hz1,
    hz2,
    resample: bool = True,
):
    xyz = _get_marker_xyz(
        path_optitrack,
        seg_number,
        marker_imu_setup[f"seg{seg_number}"]["pos_single_marker"],
    )

    if resample:
        N = xyz.shape[0]
        xs = np.arange(N, step=hz1 / hz2)
        xp = np.arange(N)

        xyz = np.hstack([np.interp(xs, xp, xyz[:, i])[:, None] for i in range(3)])
        xyz = _interp_nan_values(
            xyz, lambda arr, alpha: arr[0] * (1 - alpha) + arr[1] * alpha
        )
    # milimeters -> meters
    return xyz / 1000


def _all_equal_in_last_axis(arr):
    return np.all(np.repeat(arr[..., 0:1], arr.shape[-1], axis=-1) == arr)
