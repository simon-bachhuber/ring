import os
from pathlib import Path
import time

from _compat import unbatch_gen
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ring
from ring import ml
from ring import utils
from ring.ml import ml_utils
import wandb


def test_save_load():
    params = {"matrix": jnp.zeros((100, 100))}
    test_file = "~/params1/params.pickle"
    utils.pickle_save(params, test_file, True)
    utils.pickle_save(params, test_file, True)
    with pytest.raises(Exception):
        utils.pickle_save(params, test_file, overwrite=False)
    utils.pickle_load(test_file)

    # clean up
    os.system("rm ~/params1/params.pickle")
    os.system("rmdir ~/params1")


def test_save_load_generators():
    path = Path(__file__).parent.joinpath("gen.pickle")

    sys = ring.io.load_example("test_three_seg_seg2")
    rcmg = ring.RCMG(
        sys,
        add_X_imus=True,
        add_y_relpose=True,
    )
    data = rcmg.to_list()[0]
    rcmg.to_pickle(path)

    data_list = [jax.tree_map(lambda a: a[0], utils.pickle_load(path))]
    gen_reloaded = ring.RCMG.eager_gen_from_list(data_list, 1)
    data_reloaded = unbatch_gen(gen_reloaded)(jax.random.PRNGKey(1))

    assert ring.utils.tree_equal(data, data_reloaded)

    # clean up
    os.system(f"rm {str(path)}")


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
    wandb.init(project="TEST")

    params = {"matrix": jnp.zeros((100, 100))}
    test_file = "~/params2/params.pickle"
    n_episodes = 10
    callback = ml.callbacks.SaveParamsTrainingLoopCallback(test_file, cleanup=True)
    opt_state = None
    loop = ml.training_loop.TrainingLoop(
        jax.random.PRNGKey(1),
        generator,
        params,
        opt_state,
        step_fn,
        [ml_utils.WandbLogger()],
        [callback],
    )
    loop.run(n_episodes)
    wandb.finish()


class LogMetrices(ml.training_loop.TrainingLoopCallback):
    def __init__(self) -> None:
        self.t0 = time.time()

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: dict,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[ml.ml_utils.Logger],
        opt_state,
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
    logger = ml_utils.WandbLogger()
    n_episodes = 10
    callback = ml.callbacks.SaveParamsTrainingLoopCallback(
        test_path,
        last_n_params=2,
        track_metrices=[["mae", "seg1"], ["mae", "seg2"]],
        cleanup=True,
    )

    opt_state = None
    loop = ml.training_loop.TrainingLoop(
        jax.random.PRNGKey(1),
        generator,
        params,
        opt_state,
        step_fn,
        [logger],
        [LogMetrices(), callback],
    )
    loop.run(n_episodes)
    wandb.finish()


def test_wandb_logger():
    wandb.init(project="TEST")

    logger = ml_utils.WandbLogger()
    logger.log({"awesome_float": 1.33})
    logger.log({"awesome_array": np.array(1.0)})
    logger.log({"awesome_string": "yay"})

    root = Path(__file__).parent.parent.joinpath("test_ml_utils_assets")

    logger.log_image(str(root.joinpath("image1.png")))
    logger.log_image(str(root.joinpath("image2.png")))
    logger.log_video(str(root.joinpath("video.mp4")))
    wandb.finish()
