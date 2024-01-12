from collections import deque
from functools import partial
import itertools
import os
from pathlib import Path
import time
from typing import Callable, NamedTuple, Optional, Tuple
import warnings

import jax
import jax.numpy as jnp
import tree_utils

import wandb
import x_xy
from x_xy.utils import distribute_batchsize
from x_xy.utils import expand_batchsize
from x_xy.utils import merge_batchsize
from x_xy.utils import parse_path

from .ml_utils import Logger
from .ml_utils import MultimediaLogger
from .ml_utils import save
from .training_loop import send_kill_run_signal
from .training_loop import TrainingLoopCallback


def _build_eval_fn2(
    eval_metrices: dict[str, Tuple[Callable, Callable, Callable]],
    apply_fn,
    initial_state,
    pmap_size,
    vmap_size,
):
    @partial(jax.pmap, in_axes=(None, 0, 0))
    def pmap_vmap_apply(params, initial_state, X):
        return apply_fn(params, initial_state, X)[0]

    def eval_fn(params, X, y):
        X = expand_batchsize(X, pmap_size, vmap_size)
        yhat = pmap_vmap_apply(params, initial_state, X)
        yhat = merge_batchsize(yhat, pmap_size, vmap_size)

        values, post_reduce1 = {}, {}
        for metric_name, (metric_fn, reduce_fn1, reduce_fn2) in eval_metrices.items():
            assert (
                metric_name not in values
            ), f"The metric identitifier {metric_name} is not unique"

            reduce1_errors_fn = lambda q, qhat: reduce_fn1(
                jax.vmap(jax.vmap(metric_fn))(q, qhat)
            )
            post_reduce1_errors = jax.tree_map(reduce1_errors_fn, y, yhat)
            values.update({metric_name: jax.tree_map(reduce_fn2, post_reduce1_errors)})
            post_reduce1.update({metric_name: post_reduce1_errors})

        return values, post_reduce1

    return eval_fn


class EvalXyTrainingLoopCallback(TrainingLoopCallback):
    def __init__(
        self,
        init_apply,
        eval_metrices: dict[str, Tuple[Callable, Callable, Callable]],
        X: dict,
        y: dict,
        metric_identifier: str,
        eval_every: int = 5,
    ):
        "X, y can be batched or unbatched."
        self.X, self.y = tree_utils.to_3d_if_2d((X, y))
        del X, y
        _, initial_state = init_apply.init(jax.random.PRNGKey(1), self.X)
        batchsize = tree_utils.tree_shape(self.X)
        self.eval_fn = _build_eval_fn2(
            eval_metrices,
            init_apply.apply,
            _repeat_state(initial_state, batchsize),
            *distribute_batchsize(batchsize),
        )
        self.eval_every = eval_every
        self.metric_identifier = metric_identifier

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: dict,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[Logger],
    ):
        if self.eval_every == -1:
            return

        if (i_episode % self.eval_every) == 0:
            point_estimates, _ = self.eval_fn(params, self.X, self.y)
            self.last_metrices = {self.metric_identifier: point_estimates}
        metrices.update(self.last_metrices)


class AverageMetricesTLCB(TrainingLoopCallback):
    def __init__(self, metrices_names: list[list[str]], name: str):
        self.zoom_ins = metrices_names
        self.name = name

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: dict,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[Logger],
    ) -> None:
        value = 0
        for zoom_in in self.zoom_ins:
            value += _zoom_into_metrices(metrices, zoom_in)
        metrices.update({self.name: value / len(self.zoom_ins)})


class QueueElement(NamedTuple):
    value: float
    params: dict
    episode: int


class Queue:
    def __init__(self, maxlen: int = 1):
        self._storage: list[QueueElement] = []
        self.maxlen = maxlen

    def __len__(self) -> int:
        return len(self._storage)

    def insert(self, ele: QueueElement) -> None:
        sort = True
        if len(self) < self.maxlen:
            self._storage.append(ele)
        elif ele.value < self._storage[-1].value:
            self._storage[-1] = ele
        else:
            sort = False

        if sort:
            self._storage.sort(key=lambda ele: ele.value)

    def __iter__(self):
        return iter(self._storage)


def _zoom_into_metrices(metrices: dict, zoom_in: list[str]) -> float:
    zoomed_out = metrices
    for key in zoom_in:
        zoomed_out = zoomed_out[key]
    return float(zoomed_out)


class SaveParamsTrainingLoopCallback(TrainingLoopCallback):
    def __init__(
        self,
        path_to_file: str,
        upload: bool = True,
        last_n_params: int = 1,
        track_metrices: Optional[list[list[str]]] = None,
        track_metrices_eval_every: int = 5,
        cleanup: bool = False,
    ):
        self.path_to_file = path_to_file
        self.upload = upload
        self._queue = Queue(maxlen=last_n_params)
        self._loggers = []
        self._track_metrices = track_metrices
        self._value = 0.0
        self._cleanup = cleanup
        self._track_metrices_eval_every = track_metrices_eval_every

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: dict,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[Logger],
    ) -> None:
        if self._track_metrices is None:
            self._value -= 1.0
            value = self._value
        else:
            if (i_episode % self._track_metrices_eval_every) == 0:
                value = 0.0
                N = 0
                for combination in itertools.product(*self._track_metrices):
                    value += _zoom_into_metrices(metrices, combination)
                    N += 1
                value /= N
            else:
                # some very large loss such that it doesn't get added because
                # we have already added this parameter set
                value = 1e16

        ele = QueueElement(value, params, i_episode)
        self._queue.insert(ele)

        self._loggers = loggers

    def close(self):
        filenames = []
        for ele in self._queue:
            if len(self._queue) == 1:
                filename = parse_path(self.path_to_file, extension="pickle")
            else:
                value = "{:.2f}".format(ele.value).replace(".", ",")
                filename = parse_path(
                    self.path_to_file + f"_episode={ele.episode}_value={value}",
                    extension="pickle",
                )

            save(ele.params, filename, overwrite=True)
            if self.upload:
                multimedia_logger = _find_multimedia_logger(
                    self._loggers, raise_exception=False
                )
                if multimedia_logger is not None:
                    multimedia_logger.log_params(filename)
                else:
                    warnings.warn(
                        "Upload of parameters was requested but no `MultimediaLogger`"
                        " was found."
                    )

            filenames.append(filename)

        if self._cleanup:
            # wait for upload
            time.sleep(3)

            for filename in filenames:
                os.system(f"rm {filename}")

            # delete folder
            os.system(f"rmdir {str(Path(filename).parent)}")


def _find_multimedia_logger(
    loggers, raise_exception: bool = True
) -> MultimediaLogger | None:
    for logger in loggers:
        if isinstance(logger, MultimediaLogger):
            return logger

    if raise_exception:
        raise Exception(
            f"Neither `NeptuneLogger` nor `WandbLogger` was found in {loggers}"
        )
    else:
        return None


class LogGradsTrainingLoopCallBack(TrainingLoopCallback):
    def __init__(
        self,
        kill_if_larger: Optional[float] = None,
        consecutive_larger: int = 1,
    ) -> None:
        self.kill_if_larger = kill_if_larger
        self.consecutive_larger = consecutive_larger
        self.last_larger = deque(maxlen=consecutive_larger)

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: dict,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[Logger],
    ) -> None:
        gradient_log = {}
        for i, grads_tbp in enumerate(grads):
            grads_flat = tree_utils.batch_concat(grads_tbp, num_batch_dims=0)
            grads_max = jnp.max(jnp.abs(grads_flat))
            grads_norm = jnp.linalg.norm(grads_flat)
            if self.kill_if_larger is not None:
                if grads_norm > self.kill_if_larger:
                    self.last_larger.append(True)
                else:
                    self.last_larger.append(False)
                if all(self.last_larger):
                    send_kill_run_signal()
            gradient_log[f"grads_tbp_{i}_max"] = grads_max
            gradient_log[f"grads_tbp_{i}_l2norm"] = grads_norm

        metrices.update(gradient_log)


class NanKillRunCallback(TrainingLoopCallback):
    def __init__(
        self,
        print: bool = True,
    ) -> None:
        self.print = print

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: dict,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[Logger],
    ) -> None:
        params_fast_flat = tree_utils.batch_concat(params, num_batch_dims=0)
        params_is_nan = jnp.any(jnp.isnan(params_fast_flat))

        if params_is_nan:
            send_kill_run_signal()

        if params_is_nan and self.print:
            print(
                f"Parameters have converged to NaN at step {i_episode}. Exiting run.."
            )


class LogEpisodeTrainingLoopCallback(TrainingLoopCallback):
    def __init__(self, kill_after_episode: Optional[int] = None) -> None:
        self.kill_after_episode = kill_after_episode

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: dict,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[Logger],
    ) -> None:
        if self.kill_after_episode is not None and (
            i_episode >= self.kill_after_episode
        ):
            send_kill_run_signal()
        metrices.update({"i_episode": i_episode})


class TimingKillRunCallback(TrainingLoopCallback):
    def __init__(self, max_run_time_seconds: float) -> None:
        self.max_run_time_seconds = max_run_time_seconds

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: dict,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[Logger],
    ) -> None:
        runtime = time.time() - x_xy._TRAIN_TIMING_START
        if runtime > self.max_run_time_seconds:
            runtime_h = runtime / 3600
            print(f"Run is killed due to timing. Current runtime is {runtime_h}h.")
            send_kill_run_signal()


class WandbKillRun(TrainingLoopCallback):
    def __init__(self, stop_tag: str = "stop"):
        self.stop_tag = stop_tag

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: dict,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[Logger],
    ) -> None:
        if wandb.run is not None:
            tags = (
                wandb.Api(timeout=99)
                .run(path=f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
                .tags
            )
            if self.stop_tag in tags:
                send_kill_run_signal()


def _repeat_state(state, repeats: int):
    pmap_size, vmap_size = distribute_batchsize(repeats)
    return jax.vmap(jax.vmap(lambda _: state))(jnp.zeros((pmap_size, vmap_size)))
