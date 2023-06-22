import numpy as np


def _longest_nan_run(sequence):
    nan_runs = []
    current_run = 0
    for v in sequence:
        if np.isnan(v):
            current_run += 1
        else:
            if current_run > 0:
                nan_runs.append(current_run)
                current_run = 0
    return np.unique(nan_runs, return_counts=True)


def _nan_check(q, seg_number, hz):
    n, m = _longest_nan_run(q[:, 0])
    print(
        f"Final orientation estimate of seg{seg_number} has n-consecutive timesteps "
        f"with nan-values (at {hz} Hz) m-times: \n \t N: {n} \n \t M: {m}"
    )
    print(
        f"This equals a total of {np.sum(np.isnan(q[:, 0]))} timesteps with nan values "
        f"(at {hz} Hz)"
    )
    print(f"The total length of the sequence is {len(q)}")


def _interp_nan_values(arr, interp_fn):
    nan_values = []
    current_idx = -1
    current_run = 0
    for i in range(len(arr)):
        if np.isnan(arr[i, 0]):
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


def _slerp_nan_values(q):
    from qmt import quatInterp

    assert q.shape[-1] == 4
    q = _interp_nan_values(q, quatInterp)
    return q
