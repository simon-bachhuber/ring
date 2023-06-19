import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


def _imu_measurements_from_txt(
    path_imu,
    imu_file_prefix,
    imu_number: str,
    hz1,
    hz2,
    resample: bool = True,
    method: str = "cubic",
):
    from pathlib import Path

    df = pd.read_csv(
        Path(path_imu).joinpath(Path(imu_file_prefix + imu_number).with_suffix(".txt")),
        delimiter="\t",
        skiprows=4,
    )
    acc = df[["Acc_X", "Acc_Y", "Acc_Z"]].to_numpy()
    gyr = df[["Gyr_X", "Gyr_Y", "Gyr_Z"]].to_numpy()
    mag = df[["Mag_X", "Mag_Y", "Mag_Z"]].to_numpy()

    assert np.all(~np.isnan(acc))
    assert np.all(~np.isnan(gyr))
    assert np.all(~np.isnan(mag))

    if resample:
        N = acc.shape[0]
        xp = np.arange(N)
        xs = np.arange(N, step=hz1 / hz2)
        if method == "linear":
            acc, gyr, mag = map(
                lambda arr: np.hstack(
                    [np.interp(xs, xp, arr[:, i])[:, None] for i in range(3)]
                ),
                (acc, gyr, mag),
            )
        elif method == "cubic":
            acc, gyr, mag = map(lambda arr: CubicSpline(xp, arr)(xs), (acc, gyr, mag))
        else:
            raise NotImplementedError

    return {"acc": acc, "gyr": gyr, "mag": mag}
