from functools import partial
from typing import Callable, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tree_utils

import x_xy
from x_xy import maths
from x_xy.utils import distribute_batchsize
from x_xy.utils import expand_batchsize

from .callbacks import _repeat_state
from .callbacks import LogEpisodeTrainingLoopCallback
from .callbacks import LogGradsTrainingLoopCallBack
from .callbacks import NanKillRunCallback
from .callbacks import SaveParamsTrainingLoopCallback
from .callbacks import TimingKillRunCallback
from .callbacks import WandbKillRun
from .ml_utils import Logger
from .optimizer import make_optimizer
from .training_loop import TrainingLoop
from .training_loop import TrainingLoopCallback

default_metrices = {
    "mae_deg": (
        lambda q, qhat: maths.angle_error(q, qhat),
        lambda arr: jnp.rad2deg(jnp.mean(arr, axis=(0, 1))),
    ),
}


default_loss_fn = lambda q, qhat: maths.angle_error(q, qhat) ** 2


def _build_step_fn(
    metric_fn,
    apply_fn,
    initial_state,
    pmap_size,
    vmap_size,
    optimizer,
    tbp,
    tbp_skip: int,
    tbp_skip_keep_grads: bool,
):
    """Build step function that optimizes filter parameters based on `metric_fn`.
    `initial_state` has shape (pmap, vmap, state_dim)"""

    @partial(jax.value_and_grad, has_aux=True)
    def loss_fn(params, state, X, y):
        yhat, state = apply_fn(params, state, X)
        pipe = lambda q, qhat: jnp.mean(jax.vmap(jax.vmap(metric_fn))(q, qhat))
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

    def step_fn(params, opt_state, X, y):
        N = tree_utils.tree_shape(X, axis=-2)
        X, y = expand_batchsize((X, y), pmap_size, vmap_size)
        nonlocal initial_state

        debug_grads = []
        state = initial_state
        for i, (X_tbp, y_tbp) in enumerate(
            tree_utils.tree_split((X, y), int(N / tbp), axis=-2)
        ):
            (loss, state), grads = pmapped_loss_fn(params, state, X_tbp, y_tbp)
            debug_grads.append(grads)

            if tbp_skip > i:
                if not tbp_skip_keep_grads:
                    state = jax.lax.stop_gradient(state)
                continue
            else:
                state = jax.lax.stop_gradient(state)
            params, opt_state = apply_grads(grads, params, opt_state)

        return params, opt_state, {"loss": loss}, debug_grads

    return step_fn


key_generator, key_network = jax.random.split(jax.random.PRNGKey(0))


def train(
    generator: Callable,
    n_episodes: int,
    network: hk.TransformedWithState,
    optimizer: Optional[optax.GradientTransformation] = None,
    tbp: int = 1000,
    tbp_skip: int = 0,
    tbp_skip_keep_grads: bool = False,
    loggers: list[Logger] = [],
    callbacks: list[TrainingLoopCallback] = [],
    initial_params: Optional[dict] = None,
    key_network: jax.random.PRNGKey = key_network,
    key_generator: jax.random.PRNGKey = key_generator,
    callback_save_params: Optional[str] = None,
    callback_kill_if_grads_larger: Optional[float] = None,
    callback_kill_if_nan: bool = False,
    callback_kill_after_episode: Optional[int] = None,
    callback_kill_after_seconds: Optional[float] = None,
):
    """Trains RNNO

    Args:
        generator (Callable): output of the rcmg-module
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
    """

    # test if generator is batched..
    key = jax.random.PRNGKey(0)
    X, _ = generator(key)

    if tree_utils.tree_ndim(X) == 2:
        # .. if not then batch it
        generator = x_xy.algorithms.batch_generator(generator, 1)

    # .. now it most certainly is; Queue it for data
    X, _ = generator(key)

    batchsize = tree_utils.tree_shape(X)
    pmap_size, vmap_size = distribute_batchsize(batchsize)

    params, initial_state = network.init(
        key_network,
        X,
    )
    initial_state = _repeat_state(initial_state, batchsize)

    if initial_params is not None:
        params = initial_params
    del initial_params

    if optimizer is None:
        # TODO; hardcoded `n_steps_per_episode`
        optimizer = make_optimizer(3e-3, n_episodes, n_steps_per_episode=6)

    opt_state = optimizer.init(params)

    step_fn = _build_step_fn(
        default_loss_fn,
        network.apply,
        initial_state,
        pmap_size,
        vmap_size,
        optimizer,
        tbp=tbp,
        tbp_skip=tbp_skip,
        tbp_skip_keep_grads=tbp_skip_keep_grads,
    )

    eval_fn = _build_eval_fn(
        default_metrices, network.apply, initial_state, pmap_size, vmap_size
    )

    default_callbacks = [_DefaultEvalFnCallback(eval_fn), WandbKillRun()]

    if callback_save_params is not None:
        default_callbacks.append(SaveParamsTrainingLoopCallback(callback_save_params))

    if callback_kill_if_grads_larger is not None:
        default_callbacks.append(
            LogGradsTrainingLoopCallBack(
                callback_kill_if_grads_larger, consecutive_larger=18
            )
        )

    if callback_kill_if_nan:
        default_callbacks.append(NanKillRunCallback())

    # always log, because we also want `i_epsiode` to be logged in wandb
    default_callbacks.append(
        LogEpisodeTrainingLoopCallback(callback_kill_after_episode)
    )

    if callback_kill_after_seconds is not None:
        default_callbacks.append(TimingKillRunCallback(callback_kill_after_seconds))

    callbacks_all = default_callbacks + callbacks

    loop = TrainingLoop(
        key_generator,
        generator,
        params,
        opt_state,
        step_fn,
        loggers=loggers,
        callbacks=callbacks_all,
    )

    loop.run(n_episodes)


def _build_eval_fn(
    eval_metrices: dict[str, Tuple[Callable, Callable]],
    apply_fn,
    initial_state,
    pmap_size,
    vmap_size,
):
    """Build function that evaluates the filter performance.
    `initial_state` has shape (pmap, vmap, state_dim)"""

    def eval_fn(params, state, X, y):
        yhat, _ = apply_fn(params, state, X)

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

    def expand_then_pmap_eval_fn(params, X, y):
        X, y = expand_batchsize((X, y), pmap_size, vmap_size)
        return pmapped_eval_fn(params, initial_state, X, y)

    return expand_then_pmap_eval_fn


class _DefaultEvalFnCallback(TrainingLoopCallback):
    def __init__(self, eval_fn):
        self.eval_fn = eval_fn

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: dict,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[Logger],
    ):
        metrices.update(self.eval_fn(params, sample_eval[0], sample_eval[1]))
