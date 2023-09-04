import jax
import jax.numpy as jnp
import numpy as np

from x_xy.algorithms import RCMG_Config
from x_xy.algorithms import register_rr_joint
from x_xy.algorithms import setup_fn_randomize_joint_axes
from x_xy.io import load_example
from x_xy.subpkgs import pipeline


def test_virtual_input_joint_axes():
    register_rr_joint()

    sys = load_example("test_three_seg_seg2")
    sys = setup_fn_randomize_joint_axes(jax.random.PRNGKey(1), sys)
    joint_axes = sys.links.joint_params
    sys_rr = sys.replace(
        link_types=["rr" if l in ["ry", "rz"] else l for l in sys.link_types]
    )

    # test `load_data`
    # test `ry` / `rz`
    X, *_ = pipeline.load_data(
        sys,
        config=RCMG_Config(T=10.0),
        use_rcmg=True,
        virtual_input_joint_axes=True,
        artificial_imus=True,
    )

    np.testing.assert_allclose(
        X["seg1"]["joint_axes"], np.repeat(np.array([[0.0, 1, 0]]), 1000, axis=0)
    )
    # free joint
    np.testing.assert_allclose(
        X["seg2"]["joint_axes"], np.repeat(np.array([[1.0, 0, 0]]), 1000, axis=0)
    )
    np.testing.assert_allclose(
        X["seg3"]["joint_axes"], np.repeat(np.array([[0.0, 0, 1]]), 1000, axis=0)
    )

    # test `load_data`
    # test `rr`
    X, *_ = pipeline.load_data(
        sys_rr,
        config=RCMG_Config(T=10.0),
        use_rcmg=True,
        virtual_input_joint_axes=True,
        artificial_imus=True,
    )

    np.testing.assert_allclose(
        X["seg1"]["joint_axes"], np.repeat(joint_axes[1:2], 1000, axis=0)
    )
    # free joint
    np.testing.assert_allclose(
        X["seg2"]["joint_axes"], np.repeat(np.array([[1.0, 0, 0]]), 1000, axis=0)
    )
    np.testing.assert_allclose(
        X["seg3"]["joint_axes"], np.repeat(joint_axes[3:4], 1000, axis=0)
    )

    # test `make_generator`
    # test `ry` / `rz`
    # we can't really test behaviour for `rr` since the `make_generator` internally
    # builds a `setup_fn` that randomizes the `sys.links.joint_params` field
    X, _ = _tree_squeeze(
        pipeline.make_generator(
            RCMG_Config(T=10.0), 1, sys, virtual_input_joint_axes=True
        )[0](jax.random.PRNGKey(1))
    )

    np.testing.assert_allclose(
        X["seg1"]["joint_axes"], np.repeat(np.array([[0.0, 1, 0]]), 1000, axis=0)
    )
    # free joint
    np.testing.assert_allclose(
        X["seg2"]["joint_axes"], np.repeat(np.array([[1.0, 0, 0]]), 1000, axis=0)
    )
    np.testing.assert_allclose(
        X["seg3"]["joint_axes"], np.repeat(np.array([[0.0, 0, 1]]), 1000, axis=0)
    )


def _tree_squeeze(tree):
    return jax.tree_map(lambda arr: jnp.squeeze(arr), tree)
