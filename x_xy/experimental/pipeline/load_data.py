from typing import Optional
import warnings

import jax
import jax.numpy as jnp

import x_xy
from x_xy.subpkgs import sim2real
from x_xy.subpkgs import sys_composer
from x_xy.subpkgs.sim2real.sim2real import _crop_sequence


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
    xs,
    sys_xs,
    noisy: bool = True,
    random_s2s_ori: Optional[float] = None,
    delay=None,
    smoothen_degree=None,
    quasi_physical: bool = False,
) -> dict:
    sys_noimu, imu_attachment = sys_composer.make_sys_noimu(sys_xs)
    inv_imu_attachment = {val: key for key, val in imu_attachment.items()}
    X = {}
    N = xs.shape()
    for segment in sys_noimu.link_names:
        if segment in inv_imu_attachment:
            imu = inv_imu_attachment[segment]
            key, consume = jax.random.split(key)
            imu_measurements = x_xy.imu(
                xs.take(sys_xs.name_to_idx(imu), 1),
                sys_xs.gravity,
                sys_xs.dt,
                consume,
                noisy=noisy,
                random_s2s_ori=random_s2s_ori,
                delay=delay,
                smoothen_degree=smoothen_degree,
                quasi_physical=quasi_physical,
            )
        else:
            imu_measurements = {
                "acc": jnp.zeros(
                    (
                        N,
                        3,
                    )
                ),
                "gyr": jnp.zeros(
                    (
                        N,
                        3,
                    )
                ),
            }
        X[segment] = imu_measurements
    return X


def joint_axes_data(
    sys: x_xy.System, N: int, key: Optional[jax.Array] = None, noisy: bool = False
) -> dict[dict[str, jax.Array]]:
    "`sys` should be `sys_noimu`. `N` is number of timesteps"
    xaxis = jnp.array([1.0, 0, 0])
    yaxis = jnp.array([0.0, 1, 0])
    zaxis = jnp.array([0.0, 0, 1])
    id_to_axis = {"x": xaxis, "y": yaxis, "z": zaxis}
    X = {}

    def f(_, __, name, link_type, joint_params):
        if link_type in ["rx", "ry", "rz"]:
            joint_axes = id_to_axis[link_type[1]]
        elif link_type == "rr":
            joint_axes = joint_params
        elif link_type == "rr_imp":
            joint_axes = joint_params[:3]
        else:
            joint_axes = xaxis
        X[name] = {"joint_axes": joint_axes}

    x_xy.scan_sys(sys, f, "lll", sys.link_names, sys.link_types, sys.links.joint_params)
    X = jax.tree_map(lambda arr: jnp.repeat(arr[None], N, axis=0), X)

    if noisy:
        assert key is not None
        for name in X:
            key, c1, c2 = jax.random.split(key, 3)
            bias = x_xy.maths.quat_random(c1, maxval=jnp.deg2rad(5.0))
            noise = x_xy.maths.quat_random(c2, (N,), maxval=jnp.deg2rad(2.0))
            dist = x_xy.maths.quat_mul(noise, bias)
            X[name]["joint_axes"] = x_xy.maths.rotate(X[name]["joint_axes"], dist)
    return X


def load_data(
    sys: x_xy.System,
    config: Optional[x_xy.RCMG_Config] = None,
    exp_data: Optional[dict] = None,
    use_rcmg: bool = False,
    seed_rcmg: int = 1,
    seed_t1: int = 2,
    artificial_imus: bool = False,
    artificial_transform1: bool = False,
    artificial_random_transform1: bool = True,
    delete_global_translation_rotation: bool = False,
    randomize_global_translation_rotation: bool = False,
    scale_revolute_joint_angles: Optional[float] = None,
    project_joint_angles: bool = False,
    imu_link_names: Optional[list[str]] = None,
    rigid_imus: bool = True,
    t1: float = 0,
    t2: float | None = None,
    virtual_input_joint_axes: bool = False,
    quasi_physical: bool = False,
):
    sys_noimu, imu_attachment = sys_composer.make_sys_noimu(sys, imu_link_names)

    if (
        artificial_transform1
        or use_rcmg
        or delete_global_translation_rotation
        or randomize_global_translation_rotation
        or scale_revolute_joint_angles
    ):
        if not artificial_imus:
            warnings.warn("`artificial_imus` was overwritten to `True`")
            artificial_imus = True

    if use_rcmg:
        assert config is not None
        _, xs = x_xy.build_generator(sys, config)(jax.random.PRNGKey(seed_rcmg))
    else:
        assert exp_data is not None
        exp_data = _postprocess_exp_data(exp_data, imu_attachment=imu_attachment)
        xs = sim2real.xs_from_raw(sys, exp_data, t1, t2, eps_frame=None)

    key = jax.random.PRNGKey(seed_t1)
    if artificial_transform1:
        key, consume = jax.random.split(key)

        transform1_static = sys.links.transform1
        if artificial_random_transform1:
            transform1_static = x_xy.algorithms.generator._setup_fn_randomize_positions(
                consume, sys
            ).links.transform1

        # has no time-axis yet, so repeat in time
        transform1_static = transform1_static.batch().repeat(xs.shape())

        transform1_pos, transform2_rot = sim2real.unzip_xs(sys, xs)
        # TODO does not support `cor` function argument here; it should also pick
        # `transform1_pos` if connected to `floating-base`

        # pick `transform1_pos` only if connected to worldbody
        cond = jnp.array(sys.link_parents) == -1
        transform1 = _pick_from_transforms(cond, transform1_pos, transform1_static)
        xs = sim2real.zip_xs(sys, transform1, transform2_rot)

    if project_joint_angles:
        transform1_pos, transform2_rot = sim2real.unzip_xs(sys, xs)
        transform2_rot = sim2real.project_xs(sys, transform2_rot)
        xs = sim2real.zip_xs(sys, transform1_pos, transform2_rot)

    if delete_global_translation_rotation:
        xs = sim2real.delete_to_world_pos_rot(sys, xs)

    if randomize_global_translation_rotation:
        key, consume = jax.random.split(key)
        xs = sim2real.randomize_to_world_pos_rot(consume, sys, xs, config)

    if scale_revolute_joint_angles is not None:
        tranform1, transform2 = sim2real.unzip_xs(sys, xs)
        # here we include all positional joints, so no reason to also scale transform1
        transform2 = sim2real.scale_xs(sys, transform2, scale_revolute_joint_angles)
        xs = sim2real.zip_xs(sys, tranform1, transform2)

    N = xs.shape()
    if artificial_imus:
        key, consume = jax.random.split(key)
        X = imu_data(consume, xs, sys, quasi_physical=quasi_physical)
    else:
        rigid_flex = "imu_rigid" if rigid_imus else "imu_flex"
        X = {}
        for segment in sys_noimu.link_names:
            if segment in list(imu_attachment.values()):
                measurements = {
                    key: exp_data[segment][rigid_flex][key] for key in ["acc", "gyr"]
                }
                X[segment] = _crop_sequence(measurements, sys.dt, t1, t2)
            else:
                X[segment] = {
                    "acc": jnp.zeros(
                        (
                            N,
                            3,
                        )
                    ),
                    "gyr": jnp.zeros(
                        (
                            N,
                            3,
                        )
                    ),
                }

    if virtual_input_joint_axes:
        X_joint_axes = joint_axes_data(sys_noimu, N)
        for segment in X:
            X[segment].update(X_joint_axes[segment])

    y = x_xy.rel_pose(sys_noimu, xs, sys)

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
