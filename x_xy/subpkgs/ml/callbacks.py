from collections import deque
from functools import partial
import itertools
import os
from pathlib import Path
import time
from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import tree_utils

import wandb
from x_xy import base
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


class EvalXy2TrainingLoopCallback(TrainingLoopCallback):
    def __init__(
        self,
        exp_name: str,
        rnno_fn,
        sys_noimu,
        eval_metrices: dict[str, Tuple[Callable, Callable, Callable]],
        X: dict,
        y: dict,
        xs: base.Transform,
        sys_xs,
        metric_identifier: str,
        render_plot_metric: str,
        eval_every: int = 5,
        render_plot_every: int = 50,
        maximal_error: bool | list[bool] = True,
        plot: bool = False,
        render: bool = False,
        upload: bool = True,
        save2disk: bool = False,
        render_0th_epoch: bool = True,
        verbose: bool = True,
        show_cs: bool = False,
        show_cs_root: bool = True,
    ):
        "X, y is batched."

        network = rnno_fn(sys_noimu)
        self.sys_noimu, self.sys_xs = sys_noimu, sys_xs
        self.X, self.y, self.xs = X, y, xs
        self.plot, self.render = plot, render
        self.upload = upload
        self.save2disk = save2disk
        self.render_plot_metric = render_plot_metric
        self.maximal_error = (
            maximal_error if isinstance(maximal_error, list) else [maximal_error]
        )
        self.rnno_fn = rnno_fn
        self.path = f"~/experiments/{exp_name}"

        # delete batchsize dimension for init of state
        consume = jax.random.PRNGKey(1)
        _, initial_state = network.init(consume, X)
        batchsize = tree_utils.tree_shape(X)
        self.eval_fn = _build_eval_fn2(
            eval_metrices,
            network.apply,
            _repeat_state(initial_state, batchsize),
            *distribute_batchsize(batchsize),
        )
        self.eval_every = eval_every
        self.render_plot_every = render_plot_every
        self.metric_identifier = metric_identifier
        self.render_0th_epoch = render_0th_epoch
        self.verbose = verbose
        self.show_cs, self.show_cs_root = show_cs, show_cs_root

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: dict,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[Logger],
    ):
        self._params = params
        self._loggers = loggers
        self.i_episode = i_episode

        if self.eval_every == -1:
            return

        if (i_episode % self.eval_every) == 0:
            point_estimates, self.per_seq = self.eval_fn(params, self.X, self.y)
            self.last_metrices = {self.metric_identifier: point_estimates}
        metrices.update(self.last_metrices)

        if (i_episode % self.render_plot_every) == 0:
            if i_episode != 0 or self.render_0th_epoch:
                self._render_plot()

    def close(self):
        self._render_plot()

    def _render_plot(self):
        return
        if not self.plot and not self.render:
            return

        for maximal_error in self.maximal_error:
            reduce = jnp.argmax if maximal_error else jnp.argmin
            idx = reduce(
                jnp.mean(
                    tree_utils.batch_concat(self.per_seq[self.render_plot_metric]),
                    axis=-1,
                )
            )
            X, y, xs = tree_utils.tree_slice((self.X, self.y, self.xs), idx)

            def filename(prefix: str):
                return (
                    f"{prefix}_{self.metric_identifier}_{self.render_plot_metric}_"
                    f"idx={idx}_episode={self.i_episode}_maxError={int(maximal_error)}"
                )

            render_path = parse_path(
                self.path,
                "videos",
                filename("animation"),
                extension="mp4",
            )

            if self.verbose:
                print(f"--- EvalFnCallback {self.metric_identifier} --- ")

            """pipeline.predict(
                self.sys_noimu,
                self.rnno_fn,
                X,
                y,
                xs,
                self.sys_xs,
                self._params,
                plot=self.plot,
                render=self.render,
                render_path=render_path,
                verbose=self.verbose,
                show_cs=self.show_cs,
                show_cs_root=self.show_cs_root,
            )"""

            plot_path = parse_path(
                self.path,
                "plots",
                filename("plot"),
                extension="png",
            )
            if self.plot:
                import matplotlib.pyplot as plt

                plt.savefig(plot_path, dpi=300)
                plt.close()

            if self.upload:
                logger = _find_multimedia_logger(self._loggers)
                if self.render:
                    logger.log_video(render_path, step=self.i_episode)
                if self.plot:
                    logger.log_image(plot_path)

            if not self.save2disk:
                for path in [render_path, plot_path]:
                    if Path(path).exists():
                        os.system(f"rm {path}")


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


class SaveParamsTrainingLoopCallback(TrainingLoopCallback):
    def __init__(
        self,
        path_to_file: str,
        upload: bool = True,
        last_n_params: int = 1,
        track_metrices: Optional[list[list[str]]] = None,
        cleanup: bool = False,
    ):
        self.path_to_file = parse_path(path_to_file)
        self.upload = upload
        self._queue = Queue(maxlen=last_n_params)
        self._loggers = []
        self._track_metrices = track_metrices
        self._value = 0.0
        self._cleanup = cleanup

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
            value = 0.0
            N = 0
            for combination in itertools.product(*self._track_metrices):
                metrices_zoomedout = metrices
                for key in combination:
                    metrices_zoomedout = metrices_zoomedout[key]
                value += float(metrices_zoomedout)
                N += 1
            value /= N

        ele = QueueElement(value, params, i_episode)
        self._queue.insert(ele)

        self._loggers = loggers

    def close(self):
        filenames = []
        for ele in self._queue:
            if len(self._queue) == 1:
                filename = parse_path(self.path_to_file, extension="pickle")
            else:
                filename = parse_path(
                    self.path_to_file
                    + "_episode={}_value={:.4f}".format(ele.episode, ele.value),
                    extension="pickle",
                )

            save(ele.params, filename, overwrite=True)
            if self.upload:
                _find_multimedia_logger(self._loggers).log_params(filename)

            filenames.append(filename)

        if self._cleanup:
            # wait for upload
            time.sleep(3)

            for filename in filenames:
                os.system(f"rm {filename}")

            # delete folder
            os.system(f"rmdir {str(Path(filename).parent)}")


def _find_multimedia_logger(loggers):
    for logger in loggers:
        if isinstance(logger, MultimediaLogger):
            return logger
    raise Exception(f"Neither `NeptuneLogger` nor `WandbLogger` was found in {loggers}")


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
        runtime = time.time() - self.t0
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
