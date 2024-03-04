from collections import defaultdict
import random
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from tree_utils import PyTree

import x_xy
from x_xy import sim2real
from x_xy.algorithms.sensors import rescale_natural_units
from x_xy.subpkgs import exp
from x_xy.subpkgs import ml

_configs = {
    "hinUndHer": x_xy.MotionConfig(
        t_min=0.3,
        t_max=1.5,
        dang_max=3.0,
        delta_ang_min=0.5,
        pos_min=-1.5,
        pos_max=1.5,
        randomized_interpolation_angle=True,
        cor=True,
    ),
    "langsam": x_xy.MotionConfig(
        t_min=0.2,
        t_max=1.25,
        dang_max=2.0,
        randomized_interpolation_angle=True,
        dang_max_free_spherical=2.0,
        cdf_bins_min=1,
        cdf_bins_max=3,
        pos_min=-1.5,
        pos_max=1.5,
        cor=True,
    ),
    "standard": x_xy.MotionConfig(
        randomized_interpolation_angle=True,
        cdf_bins_min=1,
        cdf_bins_max=5,
        cor=True,
    ),
    "expFast": x_xy.MotionConfig(
        t_min=0.4,
        t_max=1.1,
        dang_max=jnp.deg2rad(180),
        delta_ang_min=jnp.deg2rad(60),
        delta_ang_max=jnp.deg2rad(110),
        pos_min=-1.5,
        pos_max=1.5,
        range_of_motion_hinge_method="sigmoid",
        randomized_interpolation_angle=True,
        cdf_bins_min=1,
        cdf_bins_max=3,
        cor=True,
    ),
    "expSlow": x_xy.MotionConfig(
        t_min=0.75,
        t_max=3.0,
        dang_min=0.1,
        dang_max=1.0,
        dang_min_free_spherical=0.1,
        delta_ang_min=0.4,
        dang_max_free_spherical=1.0,
        delta_ang_max_free_spherical=1.0,
        dpos_max=0.3,
        cor_dpos_max=0.3,
        range_of_motion_hinge_method="sigmoid",
        randomized_interpolation_angle=True,
        cdf_bins_min=1,
        cdf_bins_max=5,
        cor=True,
    ),
    "expFastNoSig": x_xy.MotionConfig(
        t_min=0.4,
        t_max=1.1,
        dang_max=jnp.deg2rad(180),
        delta_ang_min=jnp.deg2rad(60),
        delta_ang_max=jnp.deg2rad(110),
        pos_min=-1.5,
        pos_max=1.5,
        randomized_interpolation_angle=True,
        cdf_bins_min=1,
        cdf_bins_max=3,
        cor=True,
    ),
    "expSlowNoSig": x_xy.MotionConfig(
        t_min=0.75,
        t_max=3.0,
        dang_min=0.1,
        dang_max=1.0,
        dang_min_free_spherical=0.1,
        delta_ang_min=0.4,
        dang_max_free_spherical=1.0,
        delta_ang_max_free_spherical=1.0,
        dpos_max=0.3,
        cor_dpos_max=0.3,
        randomized_interpolation_angle=True,
        cdf_bins_min=1,
        cdf_bins_max=3,
        cor=True,
    ),
    "verySlow": x_xy.MotionConfig(
        t_min=1.5,
        t_max=5.0,
        dang_min=jnp.deg2rad(1),
        dang_max=jnp.deg2rad(30),
        delta_ang_min=jnp.deg2rad(20),
        dang_min_free_spherical=jnp.deg2rad(1),
        dang_max_free_spherical=jnp.deg2rad(10),
        delta_ang_min_free_spherical=jnp.deg2rad(5),
        dpos_max=0.3,
        cor_dpos_max=0.3,
        randomized_interpolation_angle=True,
        cdf_bins_min=1,
        cdf_bins_max=3,
        cor=True,
    ),
}


def load_config(config_name: str) -> x_xy.MotionConfig:
    return _configs[config_name]


_mae_metrices = {
    "mae_deg": (
        lambda q, qhat: x_xy.maths.angle_error(q, qhat),
        # back then we skipped 2500 steps which is a lot but
        # just to keep it consistent / comparable
        lambda arr: jnp.rad2deg(jnp.mean(arr[:, 2500:], axis=1)),
        jnp.mean,
    )
}


def build_experimental_validation_callback1(
    network,
    exp_id: str,
    motion_phase: str,
    interpret: dict[str, str],
    X_imus: dict[str, str],
    y_from_to_incl: dict[str, tuple[None | str, None | str, bool]],
    flex: bool = False,
    mag: bool = False,
):
    imu_key = "imu_flex" if flex else "imu_rigid"
    sensors = ["acc", "gyr"]
    if mag:
        sensors += ["mag"]

    exp_data = exp.load_data(exp_id, motion_phase)
    exp_data_interpreted = dict()
    for old, new in interpret.items():
        exp_data_interpreted[new] = exp_data[old].copy()
    exp_data = exp_data_interpreted

    X = dict()
    for new, imu in X_imus.items():
        X[new] = {s: exp_data[imu][imu_key][s] for s in sensors}

    y = dict()
    for new, (_from, _to, incl) in y_from_to_incl.items():
        assert not (_from is None and _to is None)
        if _to is None:
            assert not incl
            quat = exp_data[_from]["quat"]
        elif _from is None:
            quat = x_xy.maths.quat_inv(exp_data[_to]["quat"])
            if incl:
                quat = x_xy.maths.quat_project(quat, jnp.array([0.0, 0, 1]))[1]
        else:
            assert not incl
            quat = x_xy.maths.quat_mul(
                x_xy.maths.quat_inv(exp_data[_to]["quat"]), exp_data[_from]["quat"]
            )
        y[new] = quat

    return ml.EvalXyTrainingLoopCallback(
        network,
        _mae_metrices,
        X,
        y,
        metric_identifier=f"{exp_id}_{motion_phase}",
    )


def build_experimental_validation_callback2(
    init_apply_factory,
    sys_with_imus: x_xy.System,
    exp_id: str,
    motion_phase: str,
    jointaxes: bool = False,
    flex: bool = False,
    mag: bool = False,
    rootincl: bool = False,
    dt: bool = False,
    # (X,) -> X
    normalizer: Optional[Callable[[PyTree], PyTree]] = None,
    normalizer_names: Optional[list[str]] = None,
    natural_units: bool = False,
    X_transform=None,
):
    X, y, _ = pipeline_load_data(
        sys_with_imus,
        exp_id,
        motion_phase,
        None,
        flex,
        mag,
        jointaxes,
        rootincl,
        False,
        dt=dt,
    )

    if natural_units:
        assert (
            normalizer is None
        ), "Both `normalizer` and `natural_units` should not be used at the same time."
        X = {key: rescale_natural_units(val) for key, val in X.items()}

    if normalizer is not None:
        assert normalizer_names is not None
        # this system does not have suffix in link names, so
        # normalizer: {"seg2_2Seg": ...}
        # but, X: {"seg2": ...}
        suffix = "_" + sys_with_imus.model_name.split("_")[-1]
        dummy_X_link = X[list(X.keys())[0]]
        X_dummy = {name: dummy_X_link.copy() for name in normalizer_names}
        X = {name + suffix: X[name] for name in X}
        X_dummy.update(X)
        X_dummy = normalizer(X_dummy)
        # get ride of dummy values and remove suffix
        X = {name[: -len(suffix)]: X_dummy[name] for name in X}

    if X_transform is not None:
        X = X_transform(X)

    return ml.EvalXyTrainingLoopCallback(
        init_apply_factory(sys_with_imus.make_sys_noimu()[0]),
        _mae_metrices,
        X,
        y,
        metric_identifier=f"{sys_with_imus.model_name}_{exp_id}_{motion_phase}_"
        f"flex_{int(flex)}_mag_{int(mag)}_ja_{int(jointaxes)}",
    )
