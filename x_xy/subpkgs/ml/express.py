from typing import Optional

import jax.numpy as jnp

import x_xy
from x_xy.algorithms.custom_joints import GeneratorTrafoRandomizeJointAxes
from x_xy.algorithms.generator import transforms
from x_xy.subpkgs import exp
from x_xy.subpkgs import sys_composer
from x_xy.utils import to_list


def load_3Seg4Seg_system(anchor_3Seg=None, anchor_4Seg=None, use_rr_imp_joint=False):
    """
    4Seg:
        Four anchors : ["seg5", "seg2", "seg3", "seg4"]
        Two anchors  : ["seg5", "seg4"]
    3Seg:
        Three anchors: ["seg2", "seg3", "seg4"]
        Two anchors  : ["seg2", "seg4"]
    """
    delete_4Seg = ["seg1", "imu2", "imu3"]
    delete_3Seg = ["seg5", "imu3"]

    assert not (anchor_3Seg is None and anchor_4Seg is None)
    load = lambda *args: exp.load_sys(
        "S_06", None, *args, replace_rxyz="rr_imp" if use_rr_imp_joint else None
    )

    if anchor_3Seg is not None:
        sys_3Seg = load(anchor_3Seg, delete_3Seg)

    if anchor_4Seg is not None:
        sys_4Seg = load(anchor_4Seg, delete_4Seg)
        if anchor_3Seg is not None:
            sys_4Seg = sys_4Seg.add_prefix_suffix(suffix="_4Seg")
            sys = sys_composer.inject_system(sys_3Seg, sys_4Seg)
        else:
            sys = sys_4Seg
    else:
        sys = sys_3Seg

    return sys


def pipeline_make_generator(
    configs: x_xy.RCMG_Config | list[x_xy.RCMG_Config],
    bs_or_size: int,
    sys_data: x_xy.System | list[x_xy.System],
    sys_noimu: Optional[x_xy.System] = None,
    return_Xyx: bool = False,
    return_list: bool = False,
    seed: Optional[int] = None,
):
    "The 0-th system will be used for the relative pose / scan order."
    configs, sys_data = to_list(configs), to_list(sys_data)

    if sys_noimu is None:
        sys_noimu, _ = sys_composer.make_sys_noimu(sys_data[0])

    trafo_remove_output = x_xy.GeneratorTrafoRemoveOutputExtras()
    if return_Xyx:

        def trafo_remove_output(gen):  # noqa: F811
            def _gen(*args):
                (X, y), (key, q, x, sys) = gen(*args)
                return X, y, x

            return _gen

    def _one_generator(config, sys):
        return x_xy.GeneratorPipe(
            GeneratorTrafoRandomizeJointAxes(),
            transforms.GeneratorTrafoRandomizePositions(),
            transforms.GeneratorTrafoIMU(),
            transforms.GeneratorTrafoJointAxisSensor(sys_noimu),
            transforms.GeneratorTrafoRelPose(sys_noimu),
            x_xy.GeneratorTrafoRemoveInputExtras(sys),
            trafo_remove_output,
        )(config)

    gens = []
    for sys in sys_data:
        for config in configs:
            gens.append(_one_generator(config, sys))

    if return_list:
        assert seed is not None
        return x_xy.batch_generators_eager_to_list(gens, bs_or_size, seed)
    return x_xy.batch_generators_lazy(gens, bs_or_size)


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
