This module allows to preprocess Optical Motion Capture (OMC) data by
    
    - re-sampling IMUs and Optitrack to a common frequency,
    
    - synchronizing both systems,

    - filling in NaN values,
    
    - constructing quaternions by spanning an orthongal coordinate system.

It mainly exports a single function `dump_omc` which requires the following inputs:
```python
def dump_omc(
    # some name, only used for output filenames
    experiment_name: str,
    # path to json file that fully determines the experimental setup; see below
    path_marker_imu_setup_file: str,
    # path to .csv file
    path_optitrack_file: str,
    # path to folder that contains all imu .txt files 
    path_imu_folder: str,
    # path to some output folder; will create non-existing subfolders in the process 
    path_output_folder: str,
    # imu .txt file prefix
    imu_file_prefix: str = "MT_012102D5-000-000_00B483",
    hz_optitrack: int = 120,
    hz_imu: int = 40,
    hz_common: int = 100,
    verbose: bool = True,
)
```

---

`marker_imu_setup.json`

```json
{
    "seg1": {
        "xaxis_markers": [[2, 4], [1, 4], [2, 4], [2, 3]],
        "yaxis_markers": [[3, 4], [3, 4], [1, 2], [1, 2]],
        "position": [[-1, -1], [-1, 0], [1, 1], [1, 0]],
        "imu_rigid": "9F",
        "imu_flex": "B8",
        "pos_single_marker": 4
    },
    "seg2": {
        "xaxis_markers": [[1, 3], [1, 3], [3, 4]],
        "yaxis_markers": [[2, 3], [1, 4], [2, 3]],
        "position": [[-2, 0], [1, -1], [1, 0], [-1, 1]],
        "imu_rigid": "A3",
        "imu_flex": "84",
        "pos_single_marker": 3
    },
    "seg3": {
        "xaxis_markers": [[2, 3], [1, 3], [2, 3]],
        "yaxis_markers": [[1, 3], [3, 4], [3, 4]],
        "position": [[-0.5, 1], [-1, 0], [1, 0], [1, -1]],
        "imu_rigid": "A0",
        "imu_flex": "A2",
        "pos_single_marker": 3
    },
    "seg4": {
        "xaxis_markers": [[4, 2], [1, 2], [4, 3], [4, 2]],
        "yaxis_markers": [[1, 4], [2, 3], [1, 4], [2, 3]],
        "position": [[-1, -1], [1, 0], [1, -0.5], [-1, 0]],
        "imu_rigid": "A4",
        "imu_flex": "9B",
        "pos_single_marker": 2
    },
    "seg5": {
        "xaxis_markers": [[2, 4], [1, 4], [2, 4], [1, 3]],
        "yaxis_markers": [[3, 4], [3, 4], [1, 2], [1, 2]],
        "position": [[-1, -1], [-1, 0], [1, -2], [1, 0]],
        "imu_rigid": "99",
        "imu_flex": "85",
        "pos_single_marker": 4
    }
}
```