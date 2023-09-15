import jax.numpy as jnp
from neural_networks.logging import WandbLogger
from neural_networks.rnno import DustinExperiment
from neural_networks.rnno import rnno_v2_flags
from neural_networks.rnno import train
from neural_networks.rnno.training_loop_callbacks import EvalXy2TrainingLoopCallback
from neural_networks.rnno.training_loop_callbacks import LogEpisodeTrainingLoopCallback
from neural_networks.rnno.training_loop_callbacks import SaveParamsTrainingLoopCallback
import tree_utils
import wandb

import x_xy
from x_xy import maths
from x_xy.experimental import pipeline
from x_xy.subpkgs import exp_data
from x_xy.subpkgs import sys_composer

three_seg_seg2 = r"""
<x_xy model="three_seg_seg2">
    <options gravity="0 0 9.81" dt="0.01"/>
    <worldbody>
        <body name="seg2" joint="free">
            <body name="seg1" joint="ry" pos_min="-0.3 -0.02 -0.02" pos_max="-0.05 0.02 0.02"> # noqa: E501
                <body name="imu1" joint="frozen" pos_min="-0.25 -0.05 -0.05" pos_max="-0.05 0.05 0.05"/>
            </body>
            <body name="seg3" joint="rz" pos_min="0.05 -0.02 -0.02" pos_max="0.3 0.02 0.02">
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
sys_3Seg = exp_data.load_sys("S_06", None, "seg2", ["seg5", "imu3"])


def _make_3Seg_callbacks(rnno_fn):
    _mae_metrices = {
        "mae_deg": (
            lambda q, qhat: maths.angle_error(q, qhat),
            lambda arr: jnp.rad2deg(jnp.mean(arr[:, 2500:], axis=1)),
            jnp.mean,
        )
    }
    callbacks = []
    for motion_phase in ["fast", "slow"]:
        exp_data_dict = exp_data.load_data("S_06", motion_phase, motion_phase)
        X, y, xs = tree_utils.add_batch_dim(
            pipeline.load_data(
                sys_3Seg,
                exp_data=exp_data_dict,
            )
        )
        callbacks.append(
            EvalXy2TrainingLoopCallback(
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
    wandb.init(project="GKT", name="reproduce Neptune #50 - save params")
    logger = WandbLogger()
    config = x_xy.algorithms.RCMG_Config(
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
    gen, _ = pipeline.make_generator(config, 512, sys, sys_noimu)
    rnno_fn = lambda sys: rnno_v2_flags(sys)
    rnno = rnno_fn(sys_noimu)
    train(
        gen,
        1500,
        rnno,
        loggers=[logger],
        callbacks=[
            DustinExperiment(rnno_fn, 5, with_seg2=True),
            LogEpisodeTrainingLoopCallback(),
            SaveParamsTrainingLoopCallback("~/params/params_ry_rz.pickle"),
        ]
        + _make_3Seg_callbacks(rnno_fn),
    )


if __name__ == "__main__":
    main()
