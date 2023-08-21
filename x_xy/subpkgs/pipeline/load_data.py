import warnings
from typing import Optional

import jax
import jax.numpy as jnp

import x_xy
from x_xy.subpkgs import sim2real, sys_composer


def _postprocess_exp_data(exp_data: dict, rename: dict = {}, imu_attachment: dict = {}):
    reference = exp_data.copy()

    for name, exp_name in rename.items():
        exp_data[name] = reference[exp_name]

    reference = exp_data.copy()
    for imu, attachment in imu_attachment.items():
        exp_data[imu] = reference[attachment]

    return exp_data


def imu_data(
    key,
    x,
    sys,
    imu_attachment: dict,
    noisy: bool = True,
    random_s2s_ori: bool = False,
    delay=None,
    smoothen_degree=None,
    quasi_physical: bool = False,
) -> dict:
    X = {}
    for imu, attachment in imu_attachment.items():
        key, consume = jax.random.split(key)
        X[attachment] = x_xy.algorithms.imu(
            x.take(sys.name_to_idx(imu), 1),
            sys.gravity,
            sys.dt,
            consume,
            noisy=noisy,
            random_s2s_ori=random_s2s_ori,
            delay=delay,
            smoothen_degree=smoothen_degree,
            quasi_physical=quasi_physical,
        )
    return X


def autodetermine_imu_names(sys) -> list[str]:
    return [name for name in sys.link_names if name[:3] == "imu"]


def make_sys_noimu(sys, imu_link_names: Optional[list[str]] = None):
    if imu_link_names is None:
        imu_link_names = autodetermine_imu_names(sys)
    imu_attachment = {name: sys.parent_name(name) for name in imu_link_names}
    sys_noimu = sys_composer.delete_subsystem(sys, imu_link_names)
    return sys_noimu, imu_attachment


def load_data(
    sys: x_xy.base.System,
    config: Optional[x_xy.algorithms.RCMG_Config] = None,
    exp_data: Optional[dict] = None,
    rename_exp_data: dict = {},
    use_rcmg: bool = False,
    seed: int = 1,
    artificial_imus: bool = False,
    noisy_imus: bool = True,
    imu_delay=None,
    imu_smoothen_degree=None,
    quasi_physical=False,
    artificial_transform1: bool = False,
    artificial_random_transform1: bool = True,
    delete_global_translation_rotation: bool = False,
    scale_revolute_joint_angles: Optional[float] = None,
    imu_link_names: Optional[list[str]] = None,
    rigid_imus: bool = True,
    t1: float = 0,
    t2: float | None = None,
):
    key = jax.random.PRNGKey(seed)

    sys_noimu, imu_attachment = make_sys_noimu(sys, imu_link_names)

    if use_rcmg:
        assert config is not None
        key, consume = jax.random.split(key)
        _, xs = x_xy.algorithms.build_generator(sys, config)(consume)
        if not artificial_transform1:
            warnings.warn("`artificial_transform1` was overwritten to `True`")
            artificial_transform1 = True
    else:
        assert exp_data is not None
        exp_data = _postprocess_exp_data(exp_data, rename_exp_data, imu_attachment)
        xs = sim2real.xs_from_raw(sys, exp_data, t1, t2, eps_frame=None)

    if artificial_transform1:
        if not artificial_imus:
            warnings.warn("`artificial_imus` was overwritten to `True`")
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
        X = imu_data(
            consume,
            xs,
            sys,
            imu_attachment,
            noisy=noisy_imus,
            delay=imu_delay,
            smoothen_degree=imu_smoothen_degree,
            quasi_physical=quasi_physical,
        )
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
