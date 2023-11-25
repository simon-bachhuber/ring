import jax
import numpy as np

import x_xy
from x_xy import load_example
from x_xy import RCMG_Config
from x_xy.algorithms.custom_joints import register_rr_imp_joint
from x_xy.algorithms.custom_joints import register_rr_joint
from x_xy.algorithms.custom_joints import setup_fn_randomize_joint_axes
from x_xy.algorithms.custom_joints import setup_fn_randomize_joint_axes_primary_residual
from x_xy.algorithms.generator import transforms
from x_xy.subpkgs import sys_composer


def pipeline_load_data_X(
    sys: x_xy.System,
):
    sys_noimu, _ = sys_composer.make_sys_noimu(sys)

    gen = x_xy.GeneratorPipe(
        transforms.GeneratorTrafoJointAxisSensor(sys_noimu),
        x_xy.GeneratorTrafoRemoveOutputExtras(),
        x_xy.GeneratorTrafoRemoveInputExtras(sys),
    )(RCMG_Config(T=10.0))

    return gen(jax.random.PRNGKey(1))[0]


def test_virtual_input_joint_axes_rr_joint():
    register_rr_joint()

    sys = load_example("test_three_seg_seg2")
    sys = setup_fn_randomize_joint_axes(jax.random.PRNGKey(1), sys)
    joint_axes = sys.links.joint_params["rr"]["joint_axes"]
    sys_rr = sys.replace(
        link_types=[
            "rr" if link_type in ["ry", "rz"] else link_type
            for link_type in sys.link_types
        ]
    )

    # test `ry` / `rz`
    X = pipeline_load_data_X(sys)

    np.testing.assert_allclose(
        X["seg1"]["joint_axes"],
        np.repeat(np.array([[0.0, -1, 0]]), 1000, axis=0),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        X["seg3"]["joint_axes"],
        np.repeat(np.array([[0.0, 0, 1]]), 1000, axis=0),
        atol=1e-7,
    )

    # test `rr`
    X = pipeline_load_data_X(sys_rr)

    np.testing.assert_allclose(
        X["seg1"]["joint_axes"], np.repeat(-joint_axes[1:2], 1000, axis=0), atol=1e-7
    )
    np.testing.assert_allclose(
        X["seg3"]["joint_axes"], np.repeat(joint_axes[3:4], 1000, axis=0), atol=1e-7
    )


def test_virtual_input_joint_axes_rr_imp_joint():
    register_rr_imp_joint(RCMG_Config(T=10.0))

    sys = load_example("test_three_seg_seg2")
    sys = setup_fn_randomize_joint_axes_primary_residual(jax.random.PRNGKey(1), sys)
    joint_axes = sys.links.joint_params["rr_imp"]["joint_axes"]
    sys_rr_imp = sys.replace(
        link_types=[
            "rr_imp" if link_type in ["ry", "rz"] else link_type
            for link_type in sys.link_types
        ]
    )

    # test `rr_imp`
    X = pipeline_load_data_X(sys_rr_imp)

    np.testing.assert_allclose(
        X["seg1"]["joint_axes"],
        np.repeat(joint_axes[1:2], 1000, axis=0),
        atol=0.01,
        rtol=0.02,
    )
    np.testing.assert_allclose(
        X["seg3"]["joint_axes"],
        np.repeat(-joint_axes[3:4], 1000, axis=0),
        atol=0.002,
        rtol=0.002,
    )

    # test `make_generator`
    # we can't really test behaviour for `rr_imp` since the `make_generator` internally
    # builds a `setup_fn` that randomizes the `sys.links.joint_params` field
    X = pipeline_load_data_X(sys)

    np.testing.assert_allclose(
        X["seg1"]["joint_axes"],
        np.repeat(np.array([[0.0, -1, 0]]), 1000, axis=0),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        X["seg3"]["joint_axes"],
        np.repeat(np.array([[0.0, 0, 1]]), 1000, axis=0),
        atol=1e-7,
    )
