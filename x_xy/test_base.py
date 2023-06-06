import x_xy


def test_sys_idx_map():
    sys = x_xy.io.load_example("three_segs/three_seg_seg2")

    idx_map_l = sys.idx_map("l")
    idx_map_q = sys.idx_map("q")
    idx_map_d = sys.idx_map("d")

    for name in sys.link_names:
        assert idx_map_l[name] == sys.name_to_idx(name)

    assert idx_map_q["seg2"] == slice(0, 7)
    assert idx_map_d["seg2"] == slice(0, 6)

    assert idx_map_q["seg1"] == slice(7, 8)
    assert idx_map_d["seg1"] == slice(6, 7)

    assert idx_map_q["imu1"] == slice(8, 8)
    assert idx_map_d["imu1"] == slice(7, 7)

    assert idx_map_q["seg3"] == slice(8, 9)
    assert idx_map_d["seg3"] == slice(7, 8)

    assert idx_map_q["imu2"] == slice(9, 9)
    assert idx_map_d["imu2"] == slice(8, 8)
