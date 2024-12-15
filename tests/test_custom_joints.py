from _compat import unbatch_gen
import jax
import numpy as np
import pytest

import ring
from ring import MotionConfig
from ring.algorithms import custom_joints
from ring.algorithms.jcalc import _init_joint_params


def pipeline_load_data_X(
    sys: ring.System,
):
    sys_noimu, _ = sys.make_sys_noimu()

    gen = ring.RCMG(sys_noimu, ring.MotionConfig(T=10), add_X_jointaxes=1).to_lazy_gen(
        jit=False
    )

    return unbatch_gen(gen)(jax.random.PRNGKey(1))[0]


def _replace_ry_rz_with(sys, new_joint_type: str):
    sys_rr = sys.replace(
        link_types=[
            new_joint_type if link_type in ["ry", "rz"] else link_type
            for link_type in sys.link_types
        ]
    )
    return sys_rr


def test_virtual_input_joint_axes_rr_joint():
    sys = ring.io.load_example("test_three_seg_seg2")
    sys_rr = _replace_ry_rz_with(sys, "rr")
    sys_rr = _init_joint_params(jax.random.PRNGKey(1), sys_rr)
    joint_axes = sys_rr.links.joint_params["rr"]["joint_axes"]

    # test `ry` / `rz`
    X = pipeline_load_data_X(sys)

    np.testing.assert_allclose(
        -X["seg1"]["joint_axes"],
        np.repeat(np.array([[0.0, 1, 0]]), 1000, axis=0),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        -X["seg3"]["joint_axes"],
        np.repeat(np.array([[0.0, 0, 1]]), 1000, axis=0),
        atol=5e-7,
    )

    # test `rr`
    X = pipeline_load_data_X(sys_rr)

    np.testing.assert_allclose(
        X["seg1"]["joint_axes"],
        np.repeat(-joint_axes[1:2], 1000, axis=0),
        atol=2e-6,
        rtol=5e-4,
    )
    np.testing.assert_allclose(
        -X["seg3"]["joint_axes"],
        np.repeat(joint_axes[3:4], 1000, axis=0),
        atol=3e-7,
        rtol=2e-6,
    )


@pytest.mark.filterwarnings("ignore:The system has")
def test_virtual_input_joint_axes_rr_imp_joint():
    custom_joints.register_rr_imp_joint(MotionConfig(T=10.0), ang_max_deg=2.0)

    sys = ring.io.load_example("test_three_seg_seg2")
    sys_rr_imp = sys.change_joint_type("seg1", "rr_imp").change_joint_type(
        "seg3", "rr_imp"
    )
    sys_rr_imp = _init_joint_params(jax.random.PRNGKey(1), sys_rr_imp)
    joint_axes = sys_rr_imp.links.joint_params["rr_imp"]["joint_axes"]

    # test `rr_imp`
    X = pipeline_load_data_X(sys_rr_imp)

    np.testing.assert_allclose(
        -X["seg1"]["joint_axes"],
        np.repeat(joint_axes[1:2], 1000, axis=0),
        atol=0.01,
        rtol=0.02,
    )
    np.testing.assert_allclose(
        -X["seg3"]["joint_axes"],
        np.repeat(joint_axes[3:4], 1000, axis=0),
        atol=0.01,
        rtol=0.032,
    )

    # test `make_generator`
    # we can't really test behaviour for `rr_imp` since the `make_generator` internally
    # builds a `setup_fn` that randomizes the `sys.links.joint_params` field
    X = pipeline_load_data_X(sys)

    np.testing.assert_allclose(
        -X["seg1"]["joint_axes"],
        np.repeat(np.array([[0.0, 1, 0]]), 1000, axis=0),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        X["seg3"]["joint_axes"],
        np.repeat(np.array([[0.0, 0, -1]]), 1000, axis=0),
        atol=5e-7,
    )
