from typing import Any, NamedTuple, Optional

import jax
from jax import lax
import jax.numpy as jnp
from jax.tree_util import tree_map
import optax
from optax._src import base
from optax._src import numerics
from optax._src.transform import add_noise
from optax._src.transform import AddNoiseState


def make_optimizer(
    lr: float,
    n_episodes: int,
    n_steps_per_episode: int = 6,
    adap_clip: Optional[float] = 0.1,
    glob_clip: Optional[float] = 0.2,
    skip_large_update_max_normsq: float = 100.0,
    skip_large_update_warmup: int = 300,
    inner_opt=optax.lamb,
    cos_decay_twice: bool = False,
    scale_grads: Optional[float] = None,
    **inner_opt_kwargs,
):
    steps = n_steps_per_episode * n_episodes
    if cos_decay_twice:
        half_steps = int(steps / 2)
        schedule = optax.join_schedules(
            [
                optax.cosine_decay_schedule(lr, half_steps, 1e-2),
                optax.cosine_decay_schedule(lr * 1e-2, half_steps),
            ],
            [half_steps],
        )
    else:
        schedule = optax.cosine_decay_schedule(lr, steps, 1e-7)

    optimizer = optax.chain(
        (
            optax.scale_by_learning_rate(scale_grads, flip_sign=False)
            if scale_grads is not None
            else optax.identity()
        ),
        (
            optax.adaptive_grad_clip(adap_clip)
            if adap_clip is not None
            else optax.identity()
        ),
        (
            optax.clip_by_global_norm(glob_clip)
            if glob_clip is not None
            else optax.identity()
        ),
        inner_opt(schedule, **inner_opt_kwargs),
    )
    optimizer = skip_large_update(
        optimizer,
        skip_large_update_max_normsq,
        max_consecutive_toolarge=6 * 25,
        warmup=skip_large_update_warmup,
    )
    return optimizer


class SkipIfLargeUpdatesState(NamedTuple):
    toolarge_count: jnp.array
    count: jnp.array
    inner_state: Any
    add_noise_state: AddNoiseState


def _condition_not_toolarge(updates: base.Updates, max_norm_sq: float):
    norm_sq = jnp.sum(
        jnp.array([jnp.sum(p**2) for p in jax.tree_util.tree_leaves(updates)])
    )
    # This will also return False if `norm_sq` is NaN or Inf.
    return norm_sq < max_norm_sq


def skip_large_update(
    inner: base.GradientTransformation,
    max_norm_sq: float,
    max_consecutive_toolarge: int,
    warmup: int = 0,
    disturb_if_skip: bool = False,
    disturb_adaptive: bool = False,
    eta: float = 0.01,
    gamma: float = 0.55,
    seed: int = 0,
) -> base.GradientTransformation:
    "Also skips NaNs and Infs."
    inner = base.with_extra_args_support(inner)

    if disturb_adaptive:
        raise NotImplementedError

    add_noise_transform = add_noise(eta, gamma, seed)

    def init(params):
        return SkipIfLargeUpdatesState(
            toolarge_count=jnp.zeros([], jnp.int32),
            count=jnp.zeros([], jnp.int32),
            inner_state=inner.init(params),
            add_noise_state=add_noise_transform.init(params),
        )

    def update(updates, state: SkipIfLargeUpdatesState, params=None, **extra_args):
        inner_state = state.inner_state
        not_toolarge = _condition_not_toolarge(updates, max_norm_sq)
        toolarge_count = jnp.where(
            not_toolarge,
            jnp.zeros([], jnp.int32),
            numerics.safe_int32_increment(state.toolarge_count),
        )

        def do_update(updates):
            updates, new_inner_state = inner.update(
                updates, inner_state, params, **extra_args
            )
            return updates, new_inner_state, state.add_noise_state

        def reject_update(updates):
            if disturb_if_skip:
                updates, new_add_noise_state = add_noise_transform.update(
                    updates, state.add_noise_state, params
                )
            else:
                updates, new_add_noise_state = (
                    tree_map(jnp.zeros_like, updates),
                    state.add_noise_state,
                )
            return updates, inner_state, new_add_noise_state

        updates, new_inner_state, new_add_noise_state = lax.cond(
            jnp.logical_or(
                jnp.logical_or(not_toolarge, toolarge_count > max_consecutive_toolarge),
                state.count < warmup,
            ),
            do_update,
            reject_update,
            updates,
        )

        return updates, SkipIfLargeUpdatesState(
            toolarge_count=toolarge_count,
            count=numerics.safe_int32_increment(state.count),
            inner_state=new_inner_state,
            add_noise_state=new_add_noise_state,
        )

    return base.GradientTransformationExtraArgs(init=init, update=update)
