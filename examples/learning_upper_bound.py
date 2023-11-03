import argparse

import jax
import jax.numpy as jnp
import tree_utils

import wandb
import x_xy
from x_xy import maths
from x_xy.algorithms.custom_joints import GeneratorTrafoRandomizeJointAxes
from x_xy.algorithms.custom_joints import register_rr_imp_joint
from x_xy.algorithms.generator import transforms
from x_xy.subpkgs import exp
from x_xy.subpkgs import ml
from x_xy.subpkgs import sim2real
from x_xy.subpkgs import sys_composer
from x_xy.utils import to_list

__unique_exp_id = None

POSE_ESTIMATOR: str = "rr_rr_rr_known"


def _unique_exp_id() -> str:
    global __unique_exp_id
    if __unique_exp_id is None:
        import time

        __unique_exp_id = hash(time.time())
    return hex(__unique_exp_id)


def _load_systems(args):
    anchors_4Seg = ["seg5", "seg2", "seg3", "seg4"]
    anchors_3Seg = ["seg2", "seg3", "seg4"]
    delete_4Seg = ["seg1", "imu2", "imu3"]
    delete_3Seg = ["seg5", "imu3"]

    sys_4Seg_xyz = exp.load_sys("S_06", None, anchors_4Seg[0], delete_4Seg)
    sys_3Seg_yz = exp.load_sys("S_06", None, anchors_3Seg[0], delete_3Seg)

    systems_data = []

    for anchor_4seg in anchors_4Seg:
        sys_4Seg_rr = exp.load_sys("S_06", None, anchor_4seg, delete_4Seg, "rr_imp")

        if args.three_seg:
            suffix = "_4Seg"
            sys_4Seg_rr = sys_4Seg_rr.replace(
                link_names=[name + suffix for name in sys_4Seg_rr.link_names]
            )
            for anchor_3Seg in anchors_3Seg:
                sys_3Seg_rr = exp.load_sys(
                    "S_06", None, anchor_3Seg, delete_3Seg, "rr_imp"
                )
                sys_3Seg4Seg_rr = sys_composer.inject_system(sys_3Seg_rr, sys_4Seg_rr)
                systems_data.append(sys_3Seg4Seg_rr)
        else:
            systems_data.append(sys_4Seg_rr)
    return systems_data, sys_4Seg_xyz, sys_3Seg_yz


def _make_rnno_fn(args):
    def rnno_fn(sys):
        return ml.make_rnno(
            sys,
            200 if ml.on_cluster() else 10,
            50 if ml.on_cluster() else 10,
            link_output_dim=1,
            link_output_normalize=False,
            link_output_transform=jnp.exp,
        )

    return rnno_fn


def pipeline_make_generator(
    configs: x_xy.RCMG_Config | list[x_xy.RCMG_Config],
    bs: int,
    size: int,
    sys_data: x_xy.System | list[x_xy.System],
    return_Xyx: bool = False,
    neural_network_input: bool = True,
):
    configs, sys_data = to_list(configs), to_list(sys_data)
    sys_noimu, _ = sys_composer.make_sys_noimu(sys_data[0])

    if neural_network_input:
        gen = pipeline_make_generator(
            configs[0], 1, 1, sys_data[0], return_Xyx=False, neural_network_input=False
        )
        X, _ = gen(jax.random.PRNGKey(1))
        X_t0 = tree_utils.tree_slice(tree_utils.tree_slice(X, 0), 0)
        filter = ml.RNNOFilter(
            params=ml.load(pretrained=POSE_ESTIMATOR, pretrained_version=0)
        )
        filter.init(sys_noimu, X_t0)

        def nn_trafo(gen):
            def _gen(*args):
                (X, y), extras = gen(*args)
                yhat = tree_utils.tree_slice(
                    filter.predict(tree_utils.add_batch_dim(X)), 0
                )
                y = jax.tree_map(
                    lambda q, qhat: maths.angle_error(q, qhat)[..., None], y, yhat
                )

                unit_quats = maths.unit_quats_like(
                    yhat[sys_noimu.idx_to_name(sys_noimu.link_parents.index(0))]
                )
                for i, p in enumerate(sys_noimu.link_parents):
                    if p != -1:
                        continue
                    yhat[sys_noimu.idx_to_name(i)] = unit_quats
                X = x_xy.utils.dict_union(X, x_xy.utils.dict_to_nested(yhat, "yhat"))

                return (X, y), extras

            return _gen

    else:
        nn_trafo = lambda gen: gen

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
            nn_trafo,
            x_xy.GeneratorTrafoRemoveInputExtras(sys),
            trafo_remove_output,
        )(config)

    gens = []
    for sys in sys_data:
        for config in configs:
            gens.append(_one_generator(config, sys))

    assert (size // len(gens)) > 0, f"Batchsize too small. Must be at least {len(gens)}"

    sizes = len(gens) * [size // len(gens)]
    return x_xy.batch_generators_eager(gens, sizes, bs)


def pipeline_load_data(sys: x_xy.System, motion_phase: str):
    exp_data = exp.load_data("S_06", motion_phase)
    sys_noimu, imu_attachment = sys_composer.make_sys_noimu(sys)
    xml_str = exp.load_xml_str("S_06")
    xs = sim2real.xs_from_raw(
        sys_noimu, exp.link_name_pos_rot_data(exp_data, xml_str), qinv=True
    )
    N = xs.shape()
    X = {}
    for segment in sys_noimu.link_names:
        if segment in list(imu_attachment.values()):
            X[segment] = {
                key: exp_data[segment]["imu_rigid"][key] for key in ["acc", "gyr"]
            }
        else:
            zeros = jnp.zeros((N, 3))
            X[segment] = dict(acc=zeros, gyr=zeros)
    X_joint_axes = x_xy.joint_axes(sys_noimu, xs, sys_noimu)
    X = x_xy.utils.dict_union(X, X_joint_axes)
    y = x_xy.rel_pose(sys_noimu, xs)
    filter = ml.RNNOFilter(
        params=ml.load(pretrained=POSE_ESTIMATOR, pretrained_version=0)
    )
    filter.init(sys_noimu, tree_utils.tree_slice(X, 0))
    yhat = tree_utils.tree_slice(filter.predict(tree_utils.add_batch_dim(X)), 0)
    y = jax.tree_map(lambda q, qhat: maths.angle_error(q, qhat)[..., None], y, yhat)
    unit_quats = maths.unit_quats_like(
        yhat[sys_noimu.idx_to_name(sys_noimu.link_parents.index(0))]
    )
    anchor = sys_noimu.idx_to_name(sys_noimu.link_parents.index(-1))
    yhat[anchor] = unit_quats
    X = x_xy.utils.dict_union(X, x_xy.utils.dict_to_nested(yhat, "yhat"))
    return X, y, xs


def _loss_fn(y, yhat, weight: float):
    return jnp.where(yhat >= y, (yhat - y) ** 2, weight * (y - yhat) ** 2)


def eval_metrices_2(weight: float) -> dict:
    return {
        "error_bound_deg": (
            lambda y, yhat: _loss_fn(y, yhat, weight),
            lambda arr: jnp.rad2deg(jnp.mean(arr[:, 2000:], axis=(0, 1))),
        )
    }


def eval_metrices_3(weight: float) -> dict:
    return {
        "error_bound_deg": (
            lambda y, yhat: _loss_fn(y, yhat, weight),
            lambda arr: jnp.rad2deg(jnp.mean(arr[:, 2000:], axis=1)),
            jnp.mean,
        )
    }


def _make_3Seg_4Seg_callbacks(args):
    _, sys_4Seg, sys_3Seg = _load_systems(args)
    rnno_fn = _make_rnno_fn(args)

    callbacks = []
    for motion_phase in ["fast", "slow1"]:
        for name, sys_xs in zip(["3Seg", "4Seg"], [sys_3Seg, sys_4Seg]):
            X, y, xs = tree_utils.add_batch_dim(
                pipeline_load_data(sys_xs, motion_phase)
            )
            callbacks.append(
                ml.EvalXy2TrainingLoopCallback(
                    _unique_exp_id(),
                    rnno_fn,
                    sys_composer.make_sys_noimu(sys_xs)[0],
                    eval_metrices_3(args.weight),
                    X,
                    y,
                    xs,
                    sys_xs,
                    f"exp_{name}_{motion_phase}",
                    "error_bound_deg",
                )
            )
    return callbacks


def _make_config_callback(args, configs: dict):
    _, sys_4Seg, _ = _load_systems(args)
    rnno_fn = _make_rnno_fn(args)
    sys_noimu, _ = sys_composer.make_sys_noimu(sys_4Seg)
    key = jax.random.PRNGKey(1234)

    callbacks = []
    for name, config in configs.items():
        key, consume = jax.random.split(key)
        batchsize = 64 if ml.on_cluster() else 2
        X, y, xs = pipeline_make_generator(
            config, batchsize, batchsize, sys_4Seg, return_Xyx=True
        )(consume)
        callbacks.append(
            ml.EvalXy2TrainingLoopCallback(
                _unique_exp_id(),
                rnno_fn,
                sys_noimu,
                eval_metrices_3(args.weight),
                X,
                y,
                xs,
                sys_4Seg,
                f"rcmg_{name}",
                "error_bound_deg",
                maximal_error=[False, True],
            )
        )
    return callbacks


def main():
    parser = argparse.ArgumentParser(
        prog="upper_bound",
        description="Learn an upper bound of the error of pretrained RNNO.",
    )
    parser.add_argument("-w", "--wandb", action="store_true")
    parser.add_argument("--project", default="S06-4Seg")
    parser.add_argument("-n", "--name", default=None)
    parser.add_argument("-bs", "--batchsize", type=int, default=256)
    parser.add_argument("-s", "--size", type=int, default=4096)
    parser.add_argument("-e", "--episodes", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rcmg-add-standard", action="store_true")
    parser.add_argument("--rcmg-add-exp-slow", action="store_true")
    parser.add_argument("--rcmg-add-exp-fast", action="store_true")
    parser.add_argument("--three-seg", action="store_true")
    parser.add_argument("--weight", default=100.0, type=float)
    args = parser.parse_args()

    register_rr_imp_joint()

    configs = {
        "standard": x_xy.RCMG_Config(
            randomized_interpolation_angle=True,
            cdf_bins_min=1,
            cdf_bins_max=5,
            cor=True,
        ),
        "exp-fast": x_xy.RCMG_Config(
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
        "exp-slow": x_xy.RCMG_Config(
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

    train_configs = []
    if args.rcmg_add_standard:
        train_configs.append(configs["standard"])
    if args.rcmg_add_exp_slow:
        train_configs.append(configs["exp-slow"])
    if args.rcmg_add_exp_fast:
        train_configs.append(configs["exp-fast"])

    systems_generator, _, _ = _load_systems(args)
    sys_noimu, _ = sys_composer.make_sys_noimu(systems_generator[0])
    generator = pipeline_make_generator(
        train_configs,
        args.batchsize,
        args.size,
        systems_generator,
    )

    val_configs = ["standard", "exp-fast", "exp-slow"]
    callbacks = _make_3Seg_4Seg_callbacks(args) + _make_config_callback(
        args, {key: configs[key] for key in val_configs}
    )

    loggers = []
    config = vars(args)
    config.update(dict(unique_exp_id=_unique_exp_id()))
    if args.wandb:
        wandb.init(project=args.project, name=args.name, config=config)
        loggers.append(ml.WandbLogger())
    else:
        loggers.append(ml.MockMultimediaLogger())

    network = _make_rnno_fn(args)(sys_noimu)
    key1, key2 = jax.random.split(jax.random.PRNGKey(args.seed))

    optimizer = ml.make_optimizer(
        args.lr, args.episodes, 6, skip_large_update_max_normsq=100.0
    )
    ml.train(
        generator,
        args.episodes,
        network,
        optimizer=optimizer,
        loggers=loggers,
        callbacks=callbacks,
        key_generator=key1,
        key_network=key2,
        callback_save_params=f"~/params/{_unique_exp_id()}.pickle",
        loss_fn=(lambda y, yhat: _loss_fn(y, yhat, args.weight)),
        metrices=eval_metrices_2(args.weight),
    )


if __name__ == "__main__":
    main()
