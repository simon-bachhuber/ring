from typing import Optional, Tuple

import jax
import tree_utils

import x_xy

from .load_data import imu_data
from .load_data import joint_axes_data
from .load_data import make_sys_noimu


def _to_list(obj):
    if not isinstance(obj, list):
        return [obj]
    return obj


def make_generator(
    configs: x_xy.RCMG_Config | list[x_xy.RCMG_Config],
    bs: int,
    sys_data: x_xy.System | list[x_xy.System],
    sys_noimu: Optional[x_xy.System] = None,
    return_xs: bool = False,
    normalize: bool = False,
    randomize_positions: bool = True,
    random_s2s_ori: bool = False,
    noisy_imus: bool = True,
    quasi_physical: bool | list[bool] = False,
    smoothen_degree: Optional[int] = None,
    stochastic: bool = False,
    virtual_input_joint_axes: bool = False,
) -> Tuple[x_xy.Generator, Optional[x_xy.Normalizer]]:
    normalizer = None
    if normalize:
        gen, _ = make_generator(
            configs,
            bs,
            sys_data,
            sys_noimu,
            return_xs,
            False,
            randomize_positions,
            random_s2s_ori,
            noisy_imus,
            quasi_physical,
            smoothen_degree,
            stochastic,
            virtual_input_joint_axes,
        )
        normalizer = x_xy.make_normalizer_from_generator(gen)

    configs, sys_data = _to_list(configs), _to_list(sys_data)

    if sys_noimu is None:
        sys_noimu, _ = make_sys_noimu(sys_data[0])
    assert sys_noimu is not None

    def _make_generator(sys, config, quasi_physical: bool):
        def finalize_fn(key, q, x, sys):
            X = imu_data(
                key,
                x,
                sys,
                noisy=noisy_imus,
                random_s2s_ori=random_s2s_ori,
                quasi_physical=quasi_physical,
                smoothen_degree=smoothen_degree if not quasi_physical else None,
            )
            if virtual_input_joint_axes:
                # the outer `sys_noimu` does not get the updated joint-axes
                # so have to use the inner `sys` object
                sys_noimu_joint_axes, _ = make_sys_noimu(sys)
                N = tree_utils.tree_shape(X)
                X_joint_axes = joint_axes_data(sys_noimu_joint_axes, N)
                for segment in X:
                    X[segment].update(X_joint_axes[segment])

            y = x_xy.rel_pose(sys_noimu, x, sys)

            if normalizer is not None:
                X = normalizer(X)

            if return_xs:
                return X, y, x
            else:
                return X, y

        def setup_fn(key, sys):
            if randomize_positions:
                key, consume = jax.random.split(key)
                sys = x_xy.setup_fn_randomize_positions(consume, sys)
            # this just randomizes the joint axes, this random joint-axes
            # is only used if the joint type is `rr`
            # this is why there is no boolean `randomize_jointaxes` argument
            key, consume = jax.random.split(key)
            sys = x_xy.setup_fn_randomize_joint_axes(consume, sys)
            return sys

        return x_xy.build_generator(
            sys,
            config,
            setup_fn,
            finalize_fn,
        )

    gens = []
    for sys in sys_data:
        for config in configs:
            for qp in _to_list(quasi_physical):
                gens.append(_make_generator(sys, config, qp))

    if stochastic:
        return x_xy.batch_generator(gens, bs, stochastic=True), normalizer
    else:
        assert (
            bs // len(gens)
        ) > 0, f"Batchsize too small. Must be at least {len(gens)}"
        batchsizes = len(gens) * [bs // len(gens)]
        return x_xy.batch_generator(gens, batchsizes), normalizer


def make_offline_generator(
    configs: x_xy.RCMG_Config | list[x_xy.RCMG_Config],
    size: int,
    batchsize: int,
    sys_data: x_xy.System | list[x_xy.System],
    sys_noimu: Optional[x_xy.System] = None,
    imu_attachment: Optional[dict] = None,
    return_xs: bool = False,
    quasi_physical: bool | list[bool] = False,
    smoothen_degree: Optional[int] = None,
) -> Tuple[x_xy.Generator, Optional[x_xy.Normalizer]]:
    configs, sys_data = _to_list(configs), _to_list(sys_data)

    if sys_noimu is None:
        sys_noimu, _imu_attachment = make_sys_noimu(sys_data[0])
        if imu_attachment is None:
            imu_attachment = _imu_attachment

    assert sys_noimu is not None
    assert imu_attachment is not None

    def _make_generator(sys, config, quasi_physical: bool):
        def finalize_fn(key, q, x, sys):
            X = imu_data(
                key,
                x,
                sys,
                imu_attachment,
                quasi_physical=quasi_physical,
                smoothen_degree=smoothen_degree if not quasi_physical else None,
            )
            y = x_xy.rel_pose(sys_noimu, x, sys)

            if return_xs:
                return X, y, x
            else:
                return X, y

        def setup_fn(key, sys):
            key, consume = jax.random.split(key)
            sys = x_xy.setup_fn_randomize_positions(consume, sys)
            # this just randomizes the joint axes, this random joint-axes
            # is only used if the joint type is `rr`
            # this is why there is no boolean `randomize_jointaxes` argument
            key, consume = jax.random.split(key)
            sys = x_xy.setup_fn_randomize_joint_axes(consume, sys)
            return sys

        return x_xy.build_generator(
            sys,
            config,
            setup_fn,
            finalize_fn,
        )

    gens = []
    for sys in sys_data:
        for config in configs:
            for qp in _to_list(quasi_physical):
                gens.append(_make_generator(sys, config, qp))

    sizes = len(gens) * [size // len(gens)]
    return x_xy.offline_generator(gens, sizes, batchsize), None
