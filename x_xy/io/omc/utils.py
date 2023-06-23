import os
from pathlib import Path
from typing import Optional

from vispy.scene import TurntableCamera

import x_xy


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


def render_omc(
    sys_xml: str,
    omc_data: dict,
    elevation=30,
    distance=3,
    azimuth=5,
    filename: Optional[str] = None,
    **kwargs,
):
    sys = x_xy.io.load_sys_from_xml(sys_xml)
    transforms = x_xy.io.omc.postprocess.omc_to_xs(sys, omc_data, **kwargs)

    camera = TurntableCamera(elevation=elevation, distance=distance, azimuth=azimuth)
    if filename is not None:
        x_xy.render.animate(filename, sys, transforms, camera=camera, show_cs=True)
    else:
        x_xy.render.gui(sys, transforms, camera=camera, show_cs=True)
