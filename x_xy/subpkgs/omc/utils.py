import os
from pathlib import Path
from typing import Optional

import numpy as np
import tree
from tree_utils import PyTree


def crop_tail(signal: PyTree, hz: Optional[PyTree] = None):
    "Crop all signals to length of shortest signal."
    if hz is None:
        hz = tree.map_structure(lambda _: 1.0, signal)

    def length_in_seconds(arr, hz):
        assert arr.ndim < 3
        return len(arr) / hz

    signal_lengths = tree.map_structure(length_in_seconds, signal, hz)
    shortest_length = min(tree.flatten(signal_lengths))

    def crop(arr, hz):
        crop_tail = shortest_length * hz
        assert (
            crop_tail % 1
        ) == 0.0, f"No clean crop possible: shortest_length={shortest_length}; hz={hz}"
        crop_tail = int(crop_tail)
        return arr[:crop_tail]

    return tree.map_structure(crop, signal, hz)


def resample(
    signal: PyTree,
    hz_in: float | PyTree,
    hz_out: float | PyTree,
    quatdetect: bool = True,
) -> PyTree:
    from qmt import nanInterp
    from qmt import quatInterp
    from qmt import vecInterp

    hz_in, hz_out = tree.map_structure(float, (hz_in, hz_out))

    if isinstance(hz_in, float):
        hz_in = tree.map_structure(lambda _: hz_in, signal)
    if isinstance(hz_out, float):
        hz_out = tree.map_structure(lambda _: hz_out, signal)

    def resample_array(signal: np.ndarray, hz_in, hz_out):
        is1D = False
        if signal.ndim == 1:
            is1D = True
            signal = signal[:, None]
        assert signal.ndim == 2

        N = signal.shape[0]
        ts_out = np.arange(N, step=hz_in / hz_out)
        signal = nanInterp(signal)
        if quatdetect and signal.shape[1] == 4:
            signal = quatInterp(signal, ts_out)
        else:
            signal = vecInterp(signal, ts_out)
        if is1D:
            signal = signal[:, 0]
        return signal

    return tree.map_structure(resample_array, signal, hz_in, hz_out)


def autodetermine_imu_freq(path_imu: str) -> int:
    path_imu = Path(path_imu).expanduser()
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


def autodetermine_optitrack_freq(path_optitrack: str):
    path_optitrack = Path(path_optitrack).expanduser()

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


# could use instead of `qmt.nanInterp`
def _interp_nan_values(arr: np.ndarray, interp_fn):
    """Interpolate intermediate nan-values, and crop nan beginning and end.

    Args:
        arr (np.ndarray): NxF array.
        interp_fn: (2xF, float) -> F,
    """
    assert arr.ndim == 2

    nan_values = []
    current_idx = -1
    current_run = 0
    for i in range(len(arr)):
        if np.any(np.isnan(arr[i])):
            if current_run == 0:
                current_idx = i
            current_run += 1
        else:
            if current_run > 0:
                nan_values.append((current_idx, current_run))
                current_run = 0

    for start, length in nan_values:
        for i in range(length):
            alpha = (i + 1) / (length + 1)
            left = start - 1 if start != 0 else 0
            arr[start + i] = interp_fn(arr[[left, start + length]], alpha)

    # now `arr` has no more NaNs except if very first or very last value was NaN
    for start in range(len(arr)):
        if np.any(np.isnan(arr[start])):
            continue
        break

    for stop in range(len(arr) - 1, -1, -1):
        if np.any(np.isnan(arr[stop])):
            continue
        break

    return arr[start:stop]
