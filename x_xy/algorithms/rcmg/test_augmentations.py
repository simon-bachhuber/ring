import jax
import numpy as np

import x_xy
from x_xy.algorithms.rcmg.augmentations import (
    _draw_pos_uniform,
    setup_fn_randomize_positions,
)


def setup_fn_old(key, sys: x_xy.base.System) -> x_xy.base.System:
    def replace_pos(transforms, new_pos, name: str):
        i = sys.name_to_idx(name)
        return transforms.index_set(i, transforms[i].replace(pos=new_pos))

    ts = sys.links.transform1

    # seg 1 relative to seg2
    key, pos = _draw_pos_uniform(key, [-0.2, -0.02, -0.02], [-0.0, 0.02, 0.02])
    ts = replace_pos(ts, pos, "seg1")

    # imu1 relative to seg1
    key, pos = _draw_pos_uniform(key, [-0.25, -0.05, -0.05], [-0.05, 0.05, 0.05])
    ts = replace_pos(ts, pos, "imu1")

    # seg3 relative to seg2
    key, pos = _draw_pos_uniform(key, [0.0, -0.02, -0.02], [0.2, 0.02, 0.02])
    ts = replace_pos(ts, pos, "seg3")

    # seg4 relative to seg3
    key, pos = _draw_pos_uniform(key, [0.0, -0.02, -0.02], [0.4, 0.02, 0.02])
    ts = replace_pos(ts, pos, "seg4")

    # imu2 relative to seg3
    key, pos = _draw_pos_uniform(key, [0.05, -0.05, -0.05], [0.25, 0.05, 0.05])
    ts = replace_pos(ts, pos, "imu2")

    return sys.replace(links=sys.links.replace(transform1=ts))


def test_randomize_positions():
    key = jax.random.PRNGKey(1)
    xml_path = x_xy.io.examples_dir.joinpath("test_four_seg_seg2.xml")
    sys = x_xy.io.load_sys_from_xml(xml_path)

    # split key once more because the new logic `setup_fn_randomize_positions`
    # randomizes the position for each body even if the body has
    # no explicit `pos_min` and `pos_max` given in the xml
    # this is the case here for the body `seg2`
    # i.e. this is the split for `seg2` relative to `worldbody`
    internal_key, *_ = jax.random.split(key, 4)
    # then comes `seg1` relative to `seg2`
    pos_old = setup_fn_old(internal_key, sys).links.transform1.pos

    pos_new = setup_fn_randomize_positions(key, sys).links.transform1.pos

    np.testing.assert_array_equal(pos_old, pos_new)
