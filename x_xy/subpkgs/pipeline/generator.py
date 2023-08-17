from typing import Optional

import jax

import x_xy
from x_xy.algorithms import Normalizer, RCMG_Config
from x_xy.base import System
from x_xy.subpkgs import pipeline


def _to_list(obj):
    if not isinstance(obj, list):
        return [obj]
    return obj


def make_generator(
    configs: RCMG_Config | list[RCMG_Config],
    bs: int,
    sys_data: System | list[System],
    sys_noimu: System,
    imu_attachment: dict,
    return_xs: bool = False,
    normalizer: Optional[Normalizer] = None,
    randomize_positions: bool = True,
    random_s2s_ori: bool = False,
    noisy_imus: bool = True,
):
    configs, sys_data = _to_list(configs), _to_list(sys_data)

    def _make_generator(sys, config):
        def finalize_fn(key, q, x, sys):
            X = pipeline.imu_data(
                key,
                x,
                sys,
                imu_attachment,
                noisy=noisy_imus,
                random_s2s_ori=random_s2s_ori,
            )
            y = x_xy.algorithms.rel_pose(sys_noimu, x, sys)

            if normalizer is not None:
                X = normalizer(X)

            if return_xs:
                return X, y, x
            else:
                return X, y

        def setup_fn(key, sys):
            if randomize_positions:
                key, consume = jax.random.split(key)
                sys = x_xy.algorithms.setup_fn_randomize_positions(consume, sys)
            # this just randomizes the joint axes, this random joint-axes
            # is only used if the joint type is `rr`
            # this is why there is no boolean `randomize_jointaxes` argument
            key, consume = jax.random.split(key)
            sys = x_xy.algorithms.setup_fn_randomize_joint_axes(consume, sys)
            return sys

        return x_xy.algorithms.build_generator(
            sys,
            config,
            setup_fn,
            finalize_fn,
        )

    gens = []
    for sys in sys_data:
        for config in configs:
            gens.append(_make_generator(sys, config))

    assert (bs // len(gens)) > 0, f"Batchsize too small. Must be at least {len(gens)}"
    batchsizes = len(gens) * [bs // len(gens)]
    return x_xy.algorithms.batch_generator(gens, batchsizes)
