from datetime import datetime
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import wandb
from x_xy.subpkgs import ml
from x_xy.subpkgs.ml.callbacks import SaveParamsTrainingLoopCallback
from x_xy.subpkgs.ml.ml_utils import Logger
from x_xy.subpkgs.ml.training_loop import TrainingLoop
from x_xy.subpkgs.ml.training_loop import TrainingLoopCallback


def test_pretrained():
    exceptions = ["rr_rr_rr_known"]
    for pretrained in ml.list_pretrained():
        if pretrained in exceptions:
            continue

        ml.load(pretrained=pretrained)


def test_save_load():
    params = {"matrix": jnp.zeros((100, 100))}
    test_file = "~/params1/params.pickle"
    ml.save(params, test_file, True)
    ml.save(params, test_file, True)
    with pytest.raises(RuntimeError):
        ml.save(params, test_file, overwrite=False)
    ml.load(test_file)

    # clean up
    os.system("rm ~/params1/params.pickle")
    os.system("rmdir ~/params1")


def generator(key):
    # time, features
    X = y = jnp.zeros(
        (
            1,
            1,
        )
    )
    return X, y


def step_fn(params, opt_state, X, y):
    time.sleep(0.1)
    debug_grads = [params, params]
    return params, opt_state, {"loss": jnp.array(0.0)}, debug_grads


def test_save_params_loop_callback():
    params = {"matrix": jnp.zeros((100, 100))}
    test_file = "~/params2/params.pickle"
    logger = ml.NeptuneLogger("iss/test", name=str(datetime.now()))
    n_episodes = 10
    callback = SaveParamsTrainingLoopCallback(test_file, cleanup=True)

    opt_state = None
    loop = TrainingLoop(
        jax.random.PRNGKey(1),
        generator,
        params,
        opt_state,
        step_fn,
        [logger],
        [callback],
    )
    loop.run(n_episodes)


class LogMetrices(TrainingLoopCallback):
    def __init__(self) -> None:
        self.t0 = time.time()

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: dict,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[Logger],
    ) -> None:
        metrices.update(
            {
                "mae": {
                    "seg1": jnp.array((time.time() - self.t0) * 10),
                    "seg2": jnp.array(1.0),
                }
            }
        )


def test_save_params_metric_tracking():
    params = {"matrix": jnp.zeros((100, 100))}
    test_path = "~/params3/expID"

    wandb.init(project="TEST")
    logger = ml.WandbLogger()
    n_episodes = 10
    callback = SaveParamsTrainingLoopCallback(
        test_path,
        last_n_params=2,
        track_metrices=[["mae"], ["seg1", "seg2"]],
        cleanup=True,
    )

    opt_state = None
    loop = TrainingLoop(
        jax.random.PRNGKey(1),
        generator,
        params,
        opt_state,
        step_fn,
        [logger],
        [LogMetrices(), callback],
    )
    loop.run(n_episodes)


def test_neptune_logger():
    from datetime import datetime

    logger = ml.NeptuneLogger("iss/test", name=str(datetime.now()))
    logger.log({"awesome_float": 1.33})
    logger.log({"awesome_array": np.array(1.0)})
    logger.log({"awesome_string": "yay"})


def test_wandb_logger():
    wandb.init(project="TEST")

    logger = ml.WandbLogger()
    logger.log({"awesome_float": 1.33})
    logger.log({"awesome_array": np.array(1.0)})
    logger.log({"awesome_string": "yay"})

    root = Path(__file__).parent.parent.joinpath("test_ml_utils_assets")

    logger.log_image(str(root.joinpath("image1.png")))
    logger.log_image(str(root.joinpath("image2.png")))
    logger.log_video(str(root.joinpath("video.mp4")))
