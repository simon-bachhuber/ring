import logging
from typing import Optional

import jax
import jax.numpy as jnp
import joblib

import x_xy
from x_xy.subpkgs import sim2real, sys_composer
from x_xy.utils import parse_path


def _postprocess_exp_data(exp_data: dict, rename: dict = {}, imu_attachment: dict = {}):
    reference = exp_data.copy()

    for name, exp_name in rename.items():
        exp_data[name] = reference[exp_name]

    reference = exp_data.copy()
    for imu, attachment in imu_attachment.items():
        exp_data[imu] = reference[attachment]

    return exp_data


def _imu_data(key, x, sys, imu_attachment):
    X = {}
    for imu, attachment in imu_attachment.items():
        key, consume = jax.random.split(key)
        X[attachment] = x_xy.algorithms.imu(
            x.take(sys.name_to_idx(imu), 1), sys.gravity, sys.dt, consume, True
        )
    return X


def load_data(
    sys: x_xy.base.System,
    config: Optional[x_xy.algorithms.RCMG_Config] = None,
    path_exp_data: Optional[str] = None,
    rename_exp_data: dict = {},
    use_rcmg: bool = False,
    seed: int = 1,
    artificial_imus: bool = False,
    artificial_transform1: bool = False,
    artificial_random_transform1: bool = True,
    delete_global_translation_rotation: bool = False,
    scale_revolute_joint_angles: Optional[float] = None,
    imu_link_names: list[str] = ["imu1", "imu2"],
    rigid_imus: bool = True,
    t1: float = 0,
    t2: float | None = None,
):
    key = jax.random.PRNGKey(seed)

    imu_attachment = {name: sys.parent_name(name) for name in imu_link_names}
    sys_noimu = sys_composer.delete_subsystem(sys, imu_link_names)

    if use_rcmg:
        assert config is not None
        key, consume = jax.random.split(key)
        _, xs = x_xy.algorithms.build_generator(sys, config)(consume)
        if not artificial_transform1:
            logging.warning("`artificial_transform1` was overwritten to `True`")
            artificial_transform1 = True
    else:
        assert path_exp_data is not None
        exp_data = joblib.load(parse_path(path_exp_data, extension="joblib"))
        exp_data = _postprocess_exp_data(exp_data, rename_exp_data, imu_attachment)
        xs = sim2real.xs_from_raw(sys, exp_data, t1, t2, eps_frame=None)

    if artificial_transform1:
        if not artificial_imus:
            logging.warning("`artificial_imus` was overwritten to `True`")
            artificial_imus = True
        key, consume = jax.random.split(key)

        transform1_static = sys.links.transform1
        if artificial_random_transform1:
            transform1_static = (
                x_xy.algorithms.rcmg.augmentations.setup_fn_randomize_positions(
                    consume, sys
                ).links.transform1
            )

        # has no time-axis yet, so repeat in time
        transform1_static = transform1_static.batch().repeat(xs.shape())

        transform1_pos, transform2_rot = sim2real.unzip_xs(sys, xs)
        # pick `transform1_pos` only if connected to worldbody
        cond = jnp.array(sys.link_parents) == -1
        transform1 = _pick_from_transforms(cond, transform1_pos, transform1_static)
        xs = sim2real.zip_xs(sys, transform1, transform2_rot)

    if delete_global_translation_rotation:
        xs = sim2real.delete_to_world_pos_rot(sys, xs)

    if scale_revolute_joint_angles is not None:
        tranform1, transform2 = sim2real.unzip_xs(sys, xs)
        # here we include all positional joints, so no reason to also scale transform1
        transform2 = sim2real.scale_xs(sys, transform2, scale_revolute_joint_angles)
        xs = sim2real.zip_xs(sys, tranform1, transform2)

    if artificial_imus:
        key, consume = jax.random.split(key)
        X = _imu_data(consume, xs, sys, imu_attachment)
    else:
        rigid_flex = "imu_rigid" if rigid_imus else "imu_flex"
        X = {
            attachment: {
                key: exp_data[attachment][rigid_flex][key] for key in ["acc", "gyr"]
            }
            for attachment in imu_attachment.values()
        }
        X = sim2real._crop_sequence(X, sys.dt, t1, t2)

    y = x_xy.algorithms.rel_pose(sys_noimu, xs, sys)

    return X, y, xs


def _pick_from_transforms(cond, t1, t2):
    @jax.vmap
    def f(t1, t2):
        return jax.tree_map(
            lambda a, b: jnp.where(
                jnp.repeat(
                    cond[:, None],
                    a.shape[-1],
                    axis=-1,
                ),
                a,
                b,
            ),
            t1,
            t2,
        )

    return f(t1, t2)
