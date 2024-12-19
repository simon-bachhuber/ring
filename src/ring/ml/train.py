from functools import partial
from pathlib import Path
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
import tree_utils

from ring import maths
from ring.algorithms.generator import types
from ring.ml import base as ml_base
from ring.ml import callbacks as ml_callbacks
from ring.ml import ml_utils
from ring.ml import training_loop
from ring.utils import distribute_batchsize
from ring.utils import expand_batchsize
from ring.utils import pickle_load
import wandb

# (T, N, F) -> Scalar
LOSS_FN = Callable[[jax.Array, jax.Array], float]
_default_loss_fn = lambda q, qhat: maths.angle_error(q, qhat) ** 2

# reduces (batch_axis, time_axis) -> Scalar
ACCUMULATOR_FN = Callable[[jax.Array], float]
# Loss_fn here is: (F,) -> Scalar
METRICES = dict[str, Tuple[LOSS_FN, ACCUMULATOR_FN]]
_default_metrices = {
    "mae_deg": (
        lambda q, qhat: maths.angle_error(q, qhat),
        lambda arr: jnp.rad2deg(jnp.mean(arr, axis=(0, 1))),
    ),
}


def _build_step_fn(
    metric_fn: LOSS_FN,
    filter: ml_base.AbstractFilter,
    optimizer,
    tbp,
    skip_first_tbp_batch,
):
    """Build step function that optimizes filter parameters based on `metric_fn`.
    `initial_state` has shape (pmap, vmap, state_dim)"""

    @partial(jax.value_and_grad, has_aux=True)
    def loss_fn(params, state, X, y):
        yhat, state = filter.apply(params=params, state=state, X=X, y=y)
        # this vmap maps along batch-axis, not time-axis
        # time-axis is handled by `metric_fn`
        pipe = lambda q, qhat: jnp.mean(jax.vmap(metric_fn)(q, qhat))
        error_tree = jax.tree_map(pipe, y, yhat)
        return jnp.mean(tree_utils.batch_concat(error_tree, 0)), state

    @partial(
        jax.pmap,
        in_axes=(None, 0, 0, 0),
        out_axes=((None, 0), None),
        axis_name="devices",
    )
    def pmapped_loss_fn(params, state, X, y):
        pmean = lambda arr: jax.lax.pmean(arr, axis_name="devices")
        (loss, state), grads = loss_fn(params, state, X, y)
        return (pmean(loss), state), pmean(grads)

    @jax.jit
    def apply_grads(grads, params, opt_state):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    initial_state = None

    def step_fn(params, opt_state, X, y):
        assert X.ndim == y.ndim == 4
        B, T, N, F = X.shape
        pmap_size, vmap_size = distribute_batchsize(B)

        nonlocal initial_state
        if initial_state is None:
            initial_state = expand_batchsize(filter.init(B, X)[1], pmap_size, vmap_size)

        X, y = expand_batchsize((X, y), pmap_size, vmap_size)

        state = initial_state
        debug_grads = []
        for i, (X_tbp, y_tbp) in enumerate(
            tree_utils.tree_split((X, y), int(T / tbp), axis=-3)
        ):
            (loss, state), grads = pmapped_loss_fn(params, state, X_tbp, y_tbp)
            debug_grads.append(grads)
            if skip_first_tbp_batch and i == 0:
                continue
            state = jax.lax.stop_gradient(state)
            params, opt_state = apply_grads(grads, params, opt_state)

        return params, opt_state, {"loss": loss}, debug_grads

    return step_fn


def train_fn(
    generator: types.BatchedGenerator,
    n_episodes: int,
    filter: ml_base.AbstractFilter | ml_base.AbstractFilterWrapper,
    optimizer: Optional[optax.GradientTransformation] = optax.adam(1e-3),
    tbp: int = 1000,
    loggers: list[ml_utils.Logger] = [],
    callbacks: list[training_loop.TrainingLoopCallback] = [],
    checkpoint: Optional[str] = None,
    seed_network: int = 1,
    seed_generator: int = 2,
    callback_save_params: bool | str = False,
    callback_save_params_track_metrices: Optional[list[list[str]]] = None,
    callback_kill_if_grads_larger: Optional[float] = None,
    callback_kill_if_nan: bool = False,
    callback_kill_after_episode: Optional[int] = None,
    callback_kill_after_seconds: Optional[float] = None,
    callback_kill_tag: Optional[str] = None,
    callback_create_checkpoint: bool = True,
    loss_fn: LOSS_FN = _default_loss_fn,
    metrices: Optional[METRICES] = _default_metrices,
    link_names: Optional[list[str]] = None,
    skip_first_tbp_batch: bool = False,
) -> bool:
    """Trains RNNO

    Args:
        generator (Callable): output `build_generator`
        n_episodes (int): number of episodes to train for
        network (hk.TransformedWithState): RNNO network
        optimizer (_type_, optional): optimizer, see optimizer.py module
        tbp (int, optional): Truncated backpropagation through time step size
        tbp_skip (int, optional): Skip `tbp_skip` number of first steps per epoch.
        tbp_skip_keep_grads (bool, optional): Keeps grads between first `tbp_skip`
            steps per epoch.
        loggers: list of Loggers used to log the training progress.
        callbacks: callbacks of the TrainingLoop.
        initial_params: If given uses as initial parameters.
        key_network: PRNG Key that inits the network state and parameters.
        key_generator: PRNG Key that inits the data stream of the generator.

    Returns: bool
        Wether or not the training run was killed by a callback.
    """

    filter = filter.nojit()

    if checkpoint is not None:
        checkpoint = Path(checkpoint).with_suffix(".pickle")
        recv_checkpoint: dict = pickle_load(checkpoint)
        filter_params = recv_checkpoint["params"]
        opt_state = recv_checkpoint["opt_state"]
        del recv_checkpoint
    else:
        filter_params = filter.search_attr("params")

    if filter_params is None:
        X, _ = generator(jax.random.PRNGKey(1))
        filter_params, _ = filter.init(X=X, seed=seed_network)
        del X

    if checkpoint is None:
        opt_state = optimizer.init(filter_params)

    step_fn = _build_step_fn(
        loss_fn, filter, optimizer, tbp=tbp, skip_first_tbp_batch=skip_first_tbp_batch
    )

    # always log, because we also want `i_epsiode` to be logged in wandb
    default_callbacks = [
        ml_callbacks.LogEpisodeTrainingLoopCallback(callback_kill_after_episode)
    ]
    if metrices is not None:
        eval_fn = _build_eval_fn(metrices, filter, link_names)
        default_callbacks.append(_DefaultEvalFnCallback(eval_fn))

    if callback_kill_tag is not None:
        default_callbacks.append(ml_callbacks.WandbKillRun(stop_tag=callback_kill_tag))

    if not (callback_save_params is False):
        if callback_save_params is True:
            callback_save_params = f"~/params/{ml_utils.unique_id()}.pickle"
        default_callbacks.append(
            ml_callbacks.SaveParamsTrainingLoopCallback(callback_save_params)
        )

    if callback_kill_if_grads_larger is not None:
        default_callbacks.append(
            ml_callbacks.LogGradsTrainingLoopCallBack(
                callback_kill_if_grads_larger, consecutive_larger=18
            )
        )

    if callback_kill_if_nan:
        default_callbacks.append(ml_callbacks.NanKillRunCallback())

    if callback_kill_after_seconds is not None:
        default_callbacks.append(
            ml_callbacks.TimingKillRunCallback(callback_kill_after_seconds)
        )

    if callback_create_checkpoint:
        default_callbacks.append(ml_callbacks.CheckpointCallback())

    callbacks_all = default_callbacks + callbacks

    # we add this callback afterwards because it might require the metrices calculated
    # from one of the user-provided callbacks
    if callback_save_params_track_metrices is not None:
        assert (
            callback_save_params is not None
        ), "Required field if `callback_save_params_track_metrices` is set. Used below."

        callbacks_all.append(
            ml_callbacks.SaveParamsTrainingLoopCallback(
                path_to_file=callback_save_params,
                last_n_params=3,
                track_metrices=callback_save_params_track_metrices,
                cleanup=False,
            )
        )

    # if wandb is initialized, then add the appropriate logger
    if wandb.run is not None:
        wandb_logger_found = False
        for logger in loggers:
            if isinstance(logger, ml_utils.WandbLogger):
                wandb_logger_found = True
        if not wandb_logger_found:
            loggers.append(ml_utils.WandbLogger())

    loop = training_loop.TrainingLoop(
        jax.random.PRNGKey(seed_generator),
        generator,
        filter_params,
        opt_state,
        step_fn,
        loggers=loggers,
        callbacks=callbacks_all,
    )

    return loop.run(n_episodes)


def _arr_to_dict(y: jax.Array, link_names: list[str] | None):
    assert y.ndim == 4
    B, T, N, F = y.shape

    if link_names is None:
        link_names = ml_utils._unknown_link_names(N)

    return {name: y[..., i, :] for i, name in enumerate(link_names)}


def _build_eval_fn(
    eval_metrices: dict[str, Tuple[Callable, Callable]],
    filter: ml_base.AbstractFilter,
    link_names: Optional[list[str]] = None,
):
    """Build function that evaluates the filter performance."""

    def eval_fn(params, state, X, y):
        yhat, _ = filter.apply(params=params, state=state, X=X, y=y)

        y = _arr_to_dict(y, link_names)
        yhat = _arr_to_dict(yhat, link_names)

        values = {}
        for metric_name, (metric_fn, reduce_fn) in eval_metrices.items():
            assert (
                metric_name not in values
            ), f"The metric identitifier {metric_name} is not unique"

            pipe = lambda q, qhat: reduce_fn(jax.vmap(jax.vmap(metric_fn))(q, qhat))
            values.update({metric_name: jax.tree_map(pipe, y, yhat)})

        return values

    @partial(jax.pmap, in_axes=(None, 0, 0, 0), out_axes=None, axis_name="devices")
    def pmapped_eval_fn(params, state, X, y):
        pmean = lambda arr: jax.lax.pmean(arr, axis_name="devices")
        values = eval_fn(params, state, X, y)
        return pmean(values)

    initial_state = None

    def expand_then_pmap_eval_fn(params, X, y):
        assert X.ndim == y.ndim == 4
        B, T, N, F = X.shape
        pmap_size, vmap_size = distribute_batchsize(B)

        nonlocal initial_state
        if initial_state is None:
            initial_state = expand_batchsize(filter.init(B, X)[1], pmap_size, vmap_size)

        X, y = expand_batchsize((X, y), pmap_size, vmap_size)
        return pmapped_eval_fn(params, initial_state, X, y)

    return expand_then_pmap_eval_fn


class _DefaultEvalFnCallback(training_loop.TrainingLoopCallback):
    def __init__(self, eval_fn):
        self.eval_fn = eval_fn

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
        metrices.update(self.eval_fn(params, sample_eval[0], sample_eval[1]))
