from typing import Optional

import jax
import jax.numpy as jnp
import tree_utils

import x_xy
from x_xy.subpkgs import sys_composer
from x_xy.utils import to_list

from .load_data import imu_data
from .load_data import joint_axes_data
from .rr_imp_joint import setup_fn_randomize_joint_axes_primary_residual
from .rr_joint import setup_fn_randomize_joint_axes


def make_generator(
    configs: x_xy.RCMG_Config | list[x_xy.RCMG_Config],
    bs: int,
    sys_data: x_xy.System | list[x_xy.System],
    sys_noimu: Optional[x_xy.System] = None,
    return_xs: bool = False,
    randomize_positions: bool = True,
    random_transform1_rot_segments: Optional[float] = None,
    virtual_input_joint_axes: bool = False,
    virtual_input_joint_axes_noisy: bool = True,
    offline_size: Optional[int] = None,
) -> x_xy.algorithms.Generator:
    configs, sys_data = to_list(configs), to_list(sys_data)

    if sys_noimu is None:
        sys_noimu, _ = sys_composer.make_sys_noimu(sys_data[0])

    def _make_generator(sys, config):
        def setup_fn(key, sys):
            key, consume = jax.random.split(key)
            n_joint_params = sys.links.joint_params.shape[-1]
            if n_joint_params == 3:
                sys = setup_fn_randomize_joint_axes(consume, sys)
            elif n_joint_params == 6:
                sys = setup_fn_randomize_joint_axes_primary_residual(consume, sys)
            else:
                raise Exception("Panic :)")

            if random_transform1_rot_segments is not None:
                sys = _setup_fn_randomize_transform1_rot(
                    key, sys, random_transform1_rot_segments
                )
            return sys

        def finalize_fn(key, q, x, sys):
            key, consume = jax.random.split(key)
            X = imu_data(
                consume,
                x,
                sys,
            )
            if virtual_input_joint_axes:
                # the outer `sys_noimu` does not get the updated joint-axes
                # so have to use the inner `sys` object
                sys_noimu_joint_axes, _ = sys_composer.make_sys_noimu(sys)
                N = tree_utils.tree_shape(X)
                X_joint_axes = joint_axes_data(
                    sys_noimu_joint_axes, N, key, noisy=virtual_input_joint_axes_noisy
                )
                for segment in X:
                    X[segment].update(X_joint_axes[segment])

            y = x_xy.rel_pose(sys_noimu, x, sys)

            if return_xs:
                return X, y, x
            else:
                return X, y

        return x_xy.build_generator(
            sys,
            config,
            setup_fn,
            finalize_fn,
            randomize_positions=randomize_positions,
        )

    gens = []
    for sys in sys_data:
        for config in configs:
            gens.append(_make_generator(sys, config))

    assert (bs // len(gens)) > 0, f"Batchsize too small. Must be at least {len(gens)}"

    if offline_size is None:
        batchsizes = len(gens) * [bs // len(gens)]
        return x_xy.batch_generator(gens, batchsizes)
    else:
        sizes = len(gens) * [offline_size // len(gens)]
        return x_xy.offline_generator(gens, sizes, bs)


def _setup_fn_randomize_transform1_rot(
    key, sys, maxval, not_imus: bool = True
) -> x_xy.System:
    new_transform1 = sys.links.transform1.replace(
        rot=x_xy.maths.quat_random(key, (sys.num_links(),), maxval=maxval)
    )
    if not_imus:
        imus = [name for name in sys.link_names if name[:3] == "imu"]
        new_rot = new_transform1.rot
        for imu in imus:
            new_rot = new_rot.at[sys.name_to_idx(imu)].set(jnp.array([1.0, 0, 0, 0]))
        new_transform1 = new_transform1.replace(rot=new_rot)
    return sys.replace(links=sys.links.replace(transform1=new_transform1))
