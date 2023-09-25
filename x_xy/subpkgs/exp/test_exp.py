import pytest

from x_xy.subpkgs import exp
from x_xy.subpkgs.exp.exp import _read_yaml


def test_load_sys_data():
    exp_ids = ["S_06"]
    motions = lambda exp_id: list(_read_yaml("timings.yaml")[exp_id]["timings"].keys())[
        0:1
    ]

    for exp_id in exp_ids:
        _ = exp.load_sys(exp_id)

    for motion_start in motions(exp_id):
        _ = exp.load_data(exp_id, motion_start)


@pytest.mark.long
def test_load_sys_data_long():
    exp_ids = ["S_06"]
    motions = lambda exp_id: _read_yaml("timings.yaml")[exp_id]["timings"].keys()

    for exp_id in exp_ids:
        print(exp_id)
        for morph_key in ["seg5", "seg2", "seg3", "seg4"]:
            print(morph_key)
            for replace_with in [None, "rx"]:
                print(replace_with)
                _ = exp.load_sys(
                    exp_id, morph_yaml_key=morph_key, replace_rxyz=replace_with
                )

        for motion_start in motions(exp_id):
            print(motion_start)
            _ = exp.load_data(exp_id, motion_start)
