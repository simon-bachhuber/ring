import jax.numpy as jnp
import tree_utils

import wandb
import x_xy
from x_xy import maths
from x_xy.experimental import pipeline
from x_xy.subpkgs import exp
from x_xy.subpkgs import ml
from x_xy.subpkgs import sys_composer

pipeline.register_rr_joint()

three_seg_seg2 = r"""
<x_xy model="three_seg_seg2">
    <options gravity="0 0 9.81" dt="0.01"/>
    <worldbody>
        <body name="seg2" joint="free">
            <body name="seg1" joint="rr" pos_min="-0.3 -0.02 -0.02" pos_max="-0.05 0.02 0.02"> # noqa: E501
                <body name="imu1" joint="frozen" pos_min="-0.25 -0.05 -0.05" pos_max="-0.05 0.05 0.05"/>
            </body>
            <body name="seg3" joint="rr" pos_min="0.05 -0.02 -0.02" pos_max="0.3 0.02 0.02">
                <body name="imu2" joint="frozen" pos_min="0.05 -0.05 -0.05" pos_max="0.25 0.05 0.05"/>
            </body>
        </body>
    </worldbody>
</x_xy>
"""
sys = x_xy.io.load_sys_from_str(three_seg_seg2)
sys_noimu = sys_composer.delete_subsystem(
    sys_composer.morph_system(sys, ["seg1", -1, "seg1", "seg2", "seg3"]),
    ["imu1", "imu2"],
)
sys_3Seg = exp.load_sys("S_06", None, "seg2", ["seg5", "imu3"])


def _make_3Seg_callbacks(rnno_fn):
    _mae_metrices = {
        "mae_deg": (
            lambda q, qhat: maths.angle_error(q, qhat),
            lambda arr: jnp.rad2deg(jnp.mean(arr[:, 2000:], axis=1)),
            jnp.mean,
        )
    }
    callbacks = []
    for motion_phase in ["fast", "slow"]:
        exp_data_dict = exp.load_data("S_06", motion_phase, motion_phase)
        X, y, xs = tree_utils.add_batch_dim(
            pipeline.load_data(
                sys_3Seg,
                exp_data=exp_data_dict,
            )
        )
        callbacks.append(
            ml.EvalXy2TrainingLoopCallback(
                "repro_n50",
                rnno_fn,
                sys_composer.make_sys_noimu(sys_3Seg)[0],
                _mae_metrices,
                X,
                y,
                xs,
                sys_3Seg,
                f"exp_{motion_phase}",
                "mae_deg",
                render_plot_every=50000,
                plot=True,
                render=False,
                render_0th_epoch=False,
            )
        )
    return callbacks


def main():
    if ml.on_cluster():
        wandb.init(project="GKT", name="rr_rr_unknown")
        bs = 512
        n_episodes = 1500
        rnno_fn = lambda sys: ml.make_rnno(sys)
    else:
        bs = 4
        n_episodes = 5
        rnno_fn = lambda sys: ml.make_rnno(sys, 20, 10)

    config1 = x_xy.RCMG_Config(
        t_min=0.05,
        t_max=0.3,
        dang_min=0.1,
        dang_max=3.0,
        dpos_max=0.3,
        dang_min_free_spherical=0.1,
        dang_max_free_spherical=3.0,
        ang0_min=0.0,
        ang0_max=0.0,
    )
    config2 = x_xy.RCMG_Config(
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
    )

    gen = pipeline.make_generator([config1, config2], bs, sys, sys_noimu)

    ml.train(
        gen,
        n_episodes,
        rnno_fn(sys_noimu),
        loggers=[ml.WandbLogger() if ml.on_cluster() else ml.MockMultimediaLogger()],
        callbacks=_make_3Seg_callbacks(rnno_fn),
        callback_save_params="~/params/params_rr_rr_unknown.pickle",
    )


if __name__ == "__main__":
    main()
