from ring import exp


def test_load_sys_data():
    exp_ids = ["S_04", "S_06"]

    for exp_id in exp_ids:
        _ = exp.load_sys(exp_id)

    for motion_start in list(exp.exp.load_timings(exp_id))[:-1]:
        _ = exp.load_data(exp_id, motion_start)
