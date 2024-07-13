from _compat import _load_sys

import ring
from ring.utils import randomize_sys


def _load_1Seg2Seg3Seg4Seg_system(
    anchor_1Seg=None,
    anchor_2Seg=None,
    anchor_3Seg=None,
    anchor_4Seg=None,
):
    """
    4Seg:
        Four anchors : ["seg2", "seg3", "seg4", "seg5"]
        Two anchors  : ["seg2", "seg5"]
    3Seg:
        Three anchors: ["seg3", "seg4", "seg5"]
        Two anchors  : ["seg3", "seg5"]
    2Seg:
        Two anchors: ["seg3", "seg4"]
    1Seg:
        Single anchor: Any of the five segments.
    """
    delete_4Seg = ["seg1"]
    delete_3Seg = ["seg2"]
    delete_2Seg = ["seg2", "seg5"]
    delete_1Seg = list(
        set(["seg1", "seg2", "seg3", "seg4", "seg5"]) - set([anchor_1Seg])
    )

    assert not (
        anchor_3Seg is None
        and anchor_4Seg is None
        and anchor_2Seg is None
        and anchor_1Seg is None
    )

    def load(anchor, delete):
        sys = (
            _load_sys(1)
            .morph_system(new_anchor=anchor)
            .delete_system(delete, strict=False)
        )
        return sys

    sys = []

    if anchor_1Seg is not None:
        sys_1Seg = load(anchor_1Seg, []).change_model_name(suffix="_1Seg")
        sys.append(
            sys_1Seg.delete_system(delete_1Seg, strict=False).add_prefix_suffix(
                suffix="_1Seg"
            )
        )

    if anchor_2Seg is not None:
        sys_2Seg = (
            load(anchor_2Seg, delete_2Seg)
            .add_prefix_suffix(suffix="_2Seg")
            .change_model_name(suffix="_2Seg")
        )
        sys.append(sys_2Seg)

    if anchor_3Seg is not None:
        sys_3Seg = (
            load(anchor_3Seg, delete_3Seg)
            .add_prefix_suffix(suffix="_3Seg")
            .change_model_name(suffix="_3Seg")
        )
        sys.append(sys_3Seg)

    if anchor_4Seg is not None:
        sys_4Seg = (
            load(anchor_4Seg, delete_4Seg)
            .add_prefix_suffix(suffix="_4Seg")
            .change_model_name(suffix="_4Seg")
        )
        sys.append(sys_4Seg)

    sys_combined = sys[0]
    for other_sys in sys[1:]:
        sys_combined = sys_combined.inject_system(other_sys)

    return sys_combined


def _new_load_standard_system():
    "Generates the system `standard_sys.xml` and `standard_sys_rr_imp.xml"
    sys_4Seg: ring.System = (
        _load_sys(
            1,
        )
        .morph_system(new_anchor="seg2")
        .delete_system("seg1")
    )
    sys_3Seg = sys_4Seg.morph_system(new_anchor="seg3").delete_system("seg2")
    sys_2Seg = sys_3Seg.delete_system("seg5")
    sys_1Seg = sys_2Seg.delete_system("seg4")

    def add_suffix(sys, suffix):
        return sys.change_model_name(suffix=suffix).add_prefix_suffix(suffix=suffix)

    sys = add_suffix(sys_1Seg, "_1Seg")
    for _sys, suffix in zip(
        [sys_2Seg, sys_3Seg, sys_4Seg], ["_2Seg", "_3Seg", "_4Seg"]
    ):
        sys = sys.inject_system(add_suffix(_sys, suffix))
    return sys


def test_new_load_standard_system():
    "Test the new way of loading the 10 body standard system."

    sys = _new_load_standard_system()
    assert ring.utils.sys_compare(
        sys,
        _load_1Seg2Seg3Seg4Seg_system(
            "seg3",
            "seg3",
            "seg3",
            "seg2",
        ),
    )

    assert ring.utils.sys_compare(sys, ring.io.load_example("exclude/standard_sys"))


def SKIP_test_randomize_anchors_long():
    anchors_2Seg = ["seg3", "seg4"]
    anchors_3Seg = ["seg3", "seg5"]
    anchors_4Seg = ["seg2", "seg3", "seg4", "seg5"]

    sys_data = []
    for a2S in anchors_2Seg:
        for a3S in anchors_3Seg:
            for a4S in anchors_4Seg:
                sys_data.append(_load_1Seg2Seg3Seg4Seg_system("seg3", a2S, a3S, a4S))
    anchors = [
        "seg3_2Seg",
        "seg4_2Seg",
        "seg3_3Seg",
        "seg5_3Seg",
        "seg2_4Seg",
        "seg3_4Seg",
        "seg4_4Seg",
        "seg5_4Seg",
    ]
    sys_data_new = randomize_sys.randomize_anchors(
        ring.io.load_example("exclude/standard_sys"), anchors
    )

    for sys_old, sys_new in zip(sys_data, sys_data_new):
        assert ring.utils.sys_compare(sys_old, sys_new)


def test_randomize_anchors():
    anchors_2Seg = ["seg3", "seg4"]
    anchors_3Seg = ["seg3", "seg5"]

    sys_data = []
    for a2S in anchors_2Seg:
        for a3S in anchors_3Seg:
            sys_data.append(_load_1Seg2Seg3Seg4Seg_system(None, a2S, a3S, None))
    anchors = [
        "seg3_2Seg",
        "seg4_2Seg",
        "seg3_3Seg",
        "seg5_3Seg",
    ]
    sys_data_new = randomize_sys.randomize_anchors(sys_data[0], anchors)

    for sys_old, sys_new in zip(sys_data, sys_data_new):
        assert ring.utils.sys_compare(sys_old, sys_new)
