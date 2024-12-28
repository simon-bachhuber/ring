from collections import deque
from functools import partial
import os
from pathlib import Path
import time
from typing import Callable, NamedTuple, Optional
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import tree_utils

import ring
from ring.ml import base
from ring.ml import ml_utils
from ring.ml import training_loop
from ring.utils import distribute_batchsize
from ring.utils import expand_batchsize
from ring.utils import merge_batchsize
from ring.utils import parse_path
from ring.utils import pickle_save
import wandb


def _build_eval_fn2(
    eval_metrices: dict[str, Callable],
    filter: base.AbstractFilter,
    X: jax.Array,
    y: jax.Array,
    lam: tuple[int] | None,
    link_names: list[str] | None,
):
    filter = filter.nojit()
    assert X.ndim == 5
    assert y.ndim == 5
    y_4d = merge_batchsize(y, X.shape[0], X.shape[1])

    if link_names is None:
        link_names = ml_utils._unknown_link_names(y.shape[-2])

    @partial(jax.pmap, in_axes=(None, 0, 0))
    def pmap_vmap_apply(params, X, y):
        return filter.apply(X=X, params=params, lam=lam, y=y)[0]

    def eval_fn(params):
        yhat = pmap_vmap_apply(params, X, y)
        yhat = merge_batchsize(yhat, X.shape[0], X.shape[1])

        values = {}
        for metric_name, metric_fn in eval_metrices.items():
            assert (
                metric_name not in values
            ), f"The metric identitifier {metric_name} is not unique"
            value = jax.vmap(metric_fn, in_axes=(2, 2))(y_4d, yhat)
            assert value.ndim == 1, f"{value.shape}"
            value = {name: value[i] for i, name in enumerate(link_names)}
            values[metric_name] = value
        return values

    return eval_fn


class EvalXyTrainingLoopCallback(training_loop.TrainingLoopCallback):
    def __init__(
        self,
        filter: base.AbstractFilter,
        eval_metrices: dict[str, Callable],
        X: jax.Array,
        y: jax.Array,
        lam: tuple[int] | None,
        metric_identifier: str,
        eval_every: int = 5,
        link_names: Optional[list[str]] = None,
    ):
        """X, y can be batched or unbatched.
        Args:
            eval_metrices: "(B, T, 1) -> () and links N are vmapped."
        """
        if X.ndim == 3:
            X, y = X[None], y[None]
        B = X.shape[0]
        X, y = expand_batchsize((X, y), *distribute_batchsize(B))
        self.eval_fn = _build_eval_fn2(
            eval_metrices,
            filter,
            X,
            y,
            lam,
            link_names,
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
        loggers: list[ml_utils.Logger],
        opt_state,
    ):
        if self.eval_every == -1:
            return

        if (i_episode % self.eval_every) == 0:
            point_estimates = self.eval_fn(params)
            self.last_metrices = {self.metric_identifier: point_estimates}

        assert (
            self.metric_identifier not in metrices
        ), f"`{self.metric_identifier}` is already in `{metrices.keys()}`"
        metrices.update(self.last_metrices)


class AverageMetricesTLCB(training_loop.TrainingLoopCallback):
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
        loggers: list[ml_utils.Logger],
        opt_state,
    ) -> None:
        value = 0
        N = 0
        for zoom_in in self.zoom_ins:
            value_zoom_in = _zoom_into_metrices(metrices, zoom_in)

            if np.isnan(value_zoom_in) or np.isinf(value_zoom_in):
                warning = (
                    f"Value of zoom_in={zoom_in} is {value_zoom_in}. "
                    + f"It is not added to the metric {self.name}"
                )
                warnings.warn(warning)
                continue

            value += value_zoom_in
            N += 1

        if N > 0:
            metrices.update({self.name: value / N})


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


class SaveParamsTrainingLoopCallback(training_loop.TrainingLoopCallback):
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
        loggers: list[ml_utils.Logger | ml_utils.MixinLogger],
        opt_state,
    ) -> None:
        if self._track_metrices is None:
            self._value -= 1.0
            value = self._value
        else:
            if (i_episode % self._track_metrices_eval_every) == 0:
                value = 0.0
                N = 0
                for combination in self._track_metrices:
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
                    str(Path(self.path_to_file).with_suffix(""))
                    + f"_episode={ele.episode}_value={value}",
                    extension="pickle",
                )

            pickle_save(ele.params, filename, overwrite=False)
            if self.upload:
                success = False
                for logger in self._loggers:
                    try:
                        logger.log_params(filename)
                        success = True
                    except NotImplementedError:
                        pass
                    if not success:
                        warnings.warn(
                            "Upload of parameters was requested but no `ml_utils.Logger"
                            "` that implements `logger.log_params` was found."
                        )

            filenames.append(filename)

        if self._cleanup:
            # wait for upload
            time.sleep(3)

            for filename in filenames:
                os.system(f"rm {filename}")

            # delete folder
            os.system(f"rmdir {str(Path(filename).parent)}")


class LogGradsTrainingLoopCallBack(training_loop.TrainingLoopCallback):
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
        loggers: list[ml_utils.Logger],
        opt_state,
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
                    training_loop.send_kill_run_signal()
            gradient_log[f"grads_tbp_{i}_max"] = grads_max
            gradient_log[f"grads_tbp_{i}_l2norm"] = grads_norm

        metrices.update(gradient_log)


class NanKillRunCallback(training_loop.TrainingLoopCallback):
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
        loggers: list[ml_utils.Logger],
        opt_state,
    ) -> None:
        params_fast_flat = tree_utils.batch_concat(params, num_batch_dims=0)
        params_is_nan = jnp.any(jnp.isnan(params_fast_flat))

        if params_is_nan:
            training_loop.send_kill_run_signal()

        if params_is_nan and self.print:
            print(
                f"Parameters have converged to NaN at step {i_episode}. Exiting run.."
            )


class LogEpisodeTrainingLoopCallback(training_loop.TrainingLoopCallback):
    def __init__(self, kill_after_episode: Optional[int] = None) -> None:
        self.kill_after_episode = kill_after_episode

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: dict,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[ml_utils.Logger],
        opt_state,
    ) -> None:
        if self.kill_after_episode is not None and (
            i_episode >= self.kill_after_episode
        ):
            training_loop.send_kill_run_signal()
        metrices.update({"i_episode": i_episode})


class TimingKillRunCallback(training_loop.TrainingLoopCallback):
    def __init__(self, max_run_time_seconds: float) -> None:
        self.max_run_time_seconds = max_run_time_seconds

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: dict,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[ml_utils.Logger],
        opt_state,
    ) -> None:
        runtime = time.time() - ring._TRAIN_TIMING_START
        if runtime > self.max_run_time_seconds:
            runtime_h = runtime / 3600
            print(f"Run is killed due to timing. Current runtime is {runtime_h}h.")
            training_loop.send_kill_run_signal()


class CheckpointCallback(training_loop.TrainingLoopCallback):
    def __init__(
        self,
        checkpoint_every: Optional[int] = None,
        checkpoint_folder: str = "~/.ring_checkpoints",
    ):
        self.checkpoint_every = checkpoint_every
        self.checkpoint_folder = checkpoint_folder

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: dict,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[ml_utils.Logger],
        opt_state: tree_utils.PyTree,
    ) -> None:
        self.params = params
        self.opt_state = opt_state

        if self.checkpoint_every is not None and (
            (i_episode % self.checkpoint_every) == 0
        ):
            self._create_checkpoint()

    def _create_checkpoint(self):
        path = parse_path(
            self.checkpoint_folder, ml_utils.unique_id(), extension="pickle"
        )
        data = {"params": self.params, "opt_state": self.opt_state}
        pickle_save(
            obj=jax.device_get(data),
            path=path,
            overwrite=True,
        )

    def close(self):
        # only checkpoint if run has been killed
        if training_loop.recv_kill_run_signal():
            self._create_checkpoint()


class WandbKillRun(training_loop.TrainingLoopCallback):
    def __init__(self, stop_tag: str = "stop"):
        self.stop_tag = stop_tag

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: dict,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[ml_utils.Logger],
        opt_state,
    ) -> None:
        if wandb.run is not None:
            tags = (
                wandb.Api(timeout=99)
                .run(path=f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
                .tags
            )
            if self.stop_tag in tags:
                training_loop.send_kill_run_signal()
