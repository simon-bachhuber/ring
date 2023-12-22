from collections import defaultdict
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from tree_utils import PyTree

import x_xy
from x_xy.algorithms.sensors import rescale_natural_units
from x_xy.subpkgs import exp
from x_xy.subpkgs import ml
from x_xy.subpkgs import sim2real
from x_xy.subpkgs import sys_composer


def load_2Seg3Seg4Seg_system(
    anchor_2Seg=None,
    anchor_3Seg=None,
    anchor_4Seg=None,
    use_rr_imp=False,
    delete_inner_imus: bool = False,
    add_suffix_to_linknames: bool = False,
):
    """
    4Seg:
        Four anchors : ["seg5", "seg2", "seg3", "seg4"]
        Two anchors  : ["seg5", "seg4"]
    3Seg:
        Three anchors: ["seg2", "seg3", "seg4"]
        Two anchors  : ["seg2", "seg4"]
    2Seg:
        Two anchors: ["seg2", "seg3"]
    """
    delete_4Seg = ["seg1"]
    delete_3Seg = ["seg5"]
    delete_2Seg = ["seg5", "seg4"]

    if delete_inner_imus:
        delete_4Seg += ["imu2", "imu3"]
        delete_3Seg += ["imu3"]

    assert not (anchor_3Seg is None and anchor_4Seg is None and anchor_2Seg is None)
    load = lambda *args: exp.load_sys(
        "S_06", None, *args, replace_rxyz="rr_imp" if use_rr_imp else None
    )

    sys = []

    if anchor_2Seg is not None:
        sys_2Seg = (
            load(anchor_2Seg, delete_2Seg)
            .add_prefix_suffix(suffix="_2Seg" if add_suffix_to_linknames else None)
            .change_model_name(suffix="_2Seg")
        )
        sys.append(sys_2Seg)

    if anchor_3Seg is not None:
        sys_3Seg = (
            load(anchor_3Seg, delete_3Seg)
            .add_prefix_suffix(suffix="_3Seg" if add_suffix_to_linknames else None)
            .change_model_name(suffix="_3Seg")
        )
        sys.append(sys_3Seg)

    if anchor_4Seg is not None:
        sys_4Seg = (
            load(anchor_4Seg, delete_4Seg)
            .add_prefix_suffix(suffix="_4Seg" if add_suffix_to_linknames else None)
            .change_model_name(suffix="_4Seg")
        )
        sys.append(sys_4Seg)

    if not add_suffix_to_linknames:
        assert len(sys) == 1

    sys_combined = sys[0]
    for other_sys in sys[1:]:
        sys_combined = sys_composer.inject_system(sys_combined, other_sys)

    return sys_combined


_configs = {
    "hinUndHer": x_xy.RCMG_Config(
        t_min=0.3,
        t_max=1.5,
        dang_max=3.0,
        delta_ang_min=0.5,
        pos_min=-1.5,
        pos_max=1.5,
        randomized_interpolation_angle=True,
        cor=True,
    ),
    "langsam": x_xy.RCMG_Config(
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
    "standard": x_xy.RCMG_Config(
        randomized_interpolation_angle=True,
        cdf_bins_min=1,
        cdf_bins_max=5,
        cor=True,
    ),
    "expFast": x_xy.RCMG_Config(
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
    "expSlow": x_xy.RCMG_Config(
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
}


def load_config(config_name: str) -> x_xy.RCMG_Config:
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


def pipeline_load_data(
    sys: x_xy.System,
    exp_id: str,
    motion_phase: str,
    flex: bool,
    mag: bool,
    jointaxes: bool,
    rootincl: bool,
):
    imu_key = "imu_flex" if flex else "imu_rigid"
    sensors = ["acc", "gyr"]
    if mag:
        sensors += ["mag"]

    exp_data = exp.load_data(exp_id, motion_phase)
    sys_noimu, imu_attachment = sys_composer.make_sys_noimu(sys)
    del sys

    xs = sim2real.xs_from_raw(
        sys_noimu,
        exp.link_name_pos_rot_data(exp_data, exp.load_xml_str(exp_id)),
        qinv=True,
    )

    N = xs.shape()

    X = {}
    for segment in sys_noimu.link_names:
        if segment in list(imu_attachment.values()):
            X[segment] = {
                sensor: exp_data[segment][imu_key][sensor] for sensor in sensors
            }
        else:
            zeros = jnp.zeros((N, 3))
            X[segment] = dict(acc=zeros, gyr=zeros)

    if jointaxes:
        X_joint_axes = x_xy.joint_axes(sys_noimu, xs, sys_noimu)
        X = x_xy.utils.dict_union(X, X_joint_axes)
        # set all jointaxes to root to zero
        for name, parent in zip(sys_noimu.link_names, sys_noimu.link_parents):
            if parent == -1:
                X[name]["joint_axes"] *= 0.0

    y = x_xy.rel_pose(sys_noimu, xs)
    if rootincl:
        y_rootincl = x_xy.algorithms.sensors.root_incl(sys_noimu, xs, sys_noimu)
        y = x_xy.utils.dict_union(y, y_rootincl)

    return X, y, xs


def build_experimental_validation_callback2(
    init_apply_factory,
    sys_with_imus: x_xy.System,
    exp_id: str,
    motion_phase: str,
    jointaxes: bool = False,
    flex: bool = False,
    mag: bool = False,
    rootincl: bool = False,
    # (X,) -> X
    normalizer: Optional[Callable[[PyTree], PyTree]] = None,
    normalizer_names: Optional[list[str]] = None,
    natural_units: bool = False,
    X_transform=None,
):
    X, y, _ = pipeline_load_data(
        sys_with_imus, exp_id, motion_phase, flex, mag, jointaxes, rootincl
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
        init_apply_factory(sys_composer.make_sys_noimu(sys_with_imus)[0]),
        _mae_metrices,
        X,
        y,
        metric_identifier=f"{sys_with_imus.model_name}_{exp_id}_{motion_phase}_"
        f"flex_{int(flex)}",
    )


def rescale_natural_units_X_transform(
    X: dict[str, dict[str, jax.Array]], factor_gyr: float = 2.2, factor_ja: float = 0.57
) -> dict:
    _rescale_natural_units_fns = defaultdict(lambda: (lambda arr: arr))
    _rescale_natural_units_fns["gyr"] = lambda gyr: gyr / factor_gyr
    _rescale_natural_units_fns["acc"] = lambda acc: acc / 9.81
    _rescale_natural_units_fns["joint_axes"] = lambda arr: arr / factor_ja

    inner = lambda X: {
        key: _rescale_natural_units_fns[key](val) for key, val in X.items()
    }
    return {key: inner(val) for key, val in X.items()}
