from pathlib import Path

from x_xy.subpkgs import omc
from x_xy.utils import parse_path

HZ = 100

IMU_FILE_PREFIX1 = "MT_0120041E-000-000_00B483"
IMU_FILE_PREFIX2 = "MT_012102D5-000-000_00B483"

IMU_FILE_SETUP = {
    "S_06": (IMU_FILE_PREFIX1, ";"),
    "S_16": (IMU_FILE_PREFIX2, "\t"),
    "D_01": (IMU_FILE_PREFIX2, "\t"),
    "S_04": (IMU_FILE_PREFIX1, ";"),
}

PATH_SETUP_JSON = {"S_06": "marker_imu_setup.json"}


def exp_dir(exp_id: str) -> Path:
    return Path(__file__).parent.joinpath(exp_id)


def omc_to_joblib(
    exp_id: str, path_optitrack_csv: str, path_imu_folder: str, verbose: bool = True
):
    folder = exp_dir(exp_id)
    output_path = folder.joinpath(exp_id + ".joblib")

    imu_file_prefix, imu_file_delimiter = IMU_FILE_SETUP[exp_id]

    omc.dump_omc(
        path_marker_imu_setup_file=exp_dir(PATH_SETUP_JSON[exp_id]),
        path_optitrack_file=parse_path(path_optitrack_csv, extension="csv"),
        path_imu_folder=parse_path(path_imu_folder),
        path_output=output_path,
        imu_file_prefix=imu_file_prefix,
        imu_file_delimiter=imu_file_delimiter,
        verbose=verbose,
        hz_common=HZ,
        assume_imus_are_in_sync=True,
        save_as_matlab=False,
    )
