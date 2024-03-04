from x_xy.subpkgs import exp


def test_load_sys_data():
    exp_ids = ["S_04", "S_06"]

    for exp_id in exp_ids:
        _ = exp.load_sys(exp_id)

    for motion_start in list(exp.load_timings(exp_id))[:-1]:
        _ = exp.load_data(exp_id, motion_start)


def SKIP_test_load_sys_data_long():
    exp_ids = ["S_06"]

    for exp_id in exp_ids:
        print(exp_id)
        for morph_key in ["seg5", "seg4"]:
            print(morph_key)
            for replace_with in [None, "rx"]:
                print(replace_with)
                _ = exp.load_sys(
                    exp_id, morph_yaml_key=morph_key, replace_rxyz=replace_with
                )

        for motion_start in exp.load_timings(exp_id):
            print(motion_start)
            _ = exp.load_data(exp_id, motion_start)
