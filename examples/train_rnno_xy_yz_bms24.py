"""
Trains RNNO on a
- three-segment KC
- sparse
- rigid IMUs
- with double hinge joints with either x-/y- or y-/z-joint-axes directions.

This script creates
- the trained parameters in the path `SAVE_PARAMS_PATH`

This script requires
- about 2 hours of runtime using 4xA40 GPUs (each 48Gb VRAM)
"""

import numpy as np

import ring
from ring import ml
from ring.algorithms.generator import transforms
import wandb

# whether or not to just do a quick test run
TESTING = True

if TESTING:
    SIZE = 6
    BS = 3
    LR = 3e-3
    EPISODES = 20
    MOTION_CONFIGS = ["expFast"]
    RANDOMIZE_ANCHORS = False
    HIDDEN_STATE_DIM = 20
else:
    SIZE = 61440
    BS = 480
    LR = 3e-3
    EPISODES = 2000
    MOTION_CONFIGS = ["expFast", "expSlow", "hinUndHer", "standard"]
    RANDOMIZE_ANCHORS = True
    HIDDEN_STATE_DIM = 200

# if False train for hinge joints with y- and z-axes, else with x- and y-axes
TRAIN_XY = False
USE_WANDB = False
WANDB_PROJECT = "bms"
WANDB_NAME = f"bms-rnno-{'xy' if TRAIN_XY else 'yz'}"

SAVE_PARAMS_PATH = f"~/params/params_rnno_{'xy' if TRAIN_XY else 'yz'}.pickle"


def main():

    if USE_WANDB:
        wandb.init(project=WANDB_PROJECT, name=WANDB_NAME, config=locals())

    sys = ring.io.load_example("exclude/standard_sys").delete_system(
        ["seg3_1Seg", "seg3_2Seg", "seg2_4Seg", "imu4_3Seg"]
    )
    if TRAIN_XY:
        sys = sys.change_joint_type("seg4_3Seg", "rx", new_damp=np.array([3.0]))
        sys = sys.change_joint_type("seg5_3Seg", "ry", new_damp=np.array([3.0]))

    rnn = ring.ml.RNNO(
        12,
        return_quats=True,
        eval=False,
        hidden_state_dim=HIDDEN_STATE_DIM,
    )

    optimizer = ml.make_optimizer(
        LR,
        EPISODES,
    )

    gen = ring.RCMG(
        sys,
        [ring.MotionConfig.from_register(name) for name in MOTION_CONFIGS],
        add_X_imus=True,
        add_y_relpose=True,
        add_y_rootincl=True,
        randomize_anchors=RANDOMIZE_ANCHORS,
        randomize_positions=True,
        use_link_number_in_Xy=True,
    ).to_eager_gen(BS, SIZE)
    gen = transforms.GeneratorTrafoExpandFlatten(gen, jit=True)

    ml.train_fn(
        gen,
        EPISODES,
        rnn,
        optimizer=optimizer,
        callback_save_params=SAVE_PARAMS_PATH,
        link_names=sys.make_sys_noimu()[0].link_names,
    )


if __name__ == "__main__":
    main()
