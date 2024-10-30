from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import tree_utils

from ring.maths import safe_normalize
from ring.ml import base as ml_base
from ring.utils import pickle_load


def _scan_sys(lam: list[int], f):
    ys = []
    for i, p in enumerate(lam):
        ys.append(f(i, p))
    return tree_utils.tree_batch(ys, backend="jax")


def _make_rnno_cell_apply_fn(
    lam: list[int],
    inner_cell,
    send_msg,
    send_output,
    hidden_state_dim,
    message_dim,
    output_transform: Callable,
):
    N = len(lam)
    parent_array = jnp.array(lam, dtype=jnp.int32)

    def _rnno_cell_apply_fn(inputs, prev_state):
        empty_message = jnp.zeros((1, message_dim))
        mailbox = jnp.repeat(empty_message, N, axis=0)

        # message is sent using the hidden state of the last cell
        # for LSTM `prev_state` is of shape (2 * hidden_state_dim) du to cell state
        prev_last_hidden_state = prev_state[:, -1, :hidden_state_dim]

        msg = jnp.concatenate(
            (jax.vmap(send_msg)(prev_last_hidden_state), empty_message)
        )

        def accumulate_message(link):
            return jnp.sum(
                jnp.where(
                    jnp.repeat((parent_array == link)[:, None], message_dim, axis=-1),
                    msg[:-1],
                    mailbox,
                ),
                axis=0,
            )

        mailbox = jax.vmap(accumulate_message)(jnp.arange(N))

        def cell_input(i: int, p: int):
            local_input = inputs[i]
            local_cell_input = tree_utils.batch_concat_acme(
                (local_input, msg[p], mailbox[i]), num_batch_dims=0
            )
            return local_cell_input

        stacked_cell_input = _scan_sys(lam, cell_input)

        def update_state(cell_input, state):
            cell_output, state = inner_cell(cell_input, state)
            output = output_transform(send_output(cell_output))
            return output, state

        y, state = jax.vmap(update_state)(stacked_cell_input, prev_state)
        return y, state

    return _rnno_cell_apply_fn


def make_ring(
    lam: list[int],
    hidden_state_dim: int = 400,
    message_dim: int = 200,
    celltype: str = "gru",
    stack_rnn_cells: int = 2,
    send_message_n_layers: int = 1,
    link_output_dim: int = 4,
    link_output_normalize: bool = True,
    link_output_transform: Optional[Callable] = None,
    layernorm: bool = True,
    layernorm_trainable: bool = True,
) -> SimpleNamespace:

    if link_output_normalize:
        assert link_output_transform is None
        link_output_transform = safe_normalize
    else:
        if link_output_transform is None:
            link_output_transform = lambda x: x

    @hk.without_apply_rng
    @hk.transform_with_state
    def forward(X):
        send_msg = hk.nets.MLP(
            [hidden_state_dim] * send_message_n_layers + [message_dim]
        )

        inner_cell = StackedRNNCell(
            celltype,
            hidden_state_dim,
            stack_rnn_cells,
            layernorm=layernorm,
            layernorm_trainable=layernorm_trainable,
        )
        send_output = hk.nets.MLP([hidden_state_dim, link_output_dim])
        state = hk.get_state(
            "inner_cell_state",
            [
                len(lam),
                stack_rnn_cells,
                (hidden_state_dim * 2 if celltype == "lstm" else hidden_state_dim),
            ],
            init=jnp.zeros,
        )

        y, state = hk.dynamic_unroll(
            _make_rnno_cell_apply_fn(
                lam=lam,
                inner_cell=inner_cell,
                send_msg=send_msg,
                send_output=send_output,
                hidden_state_dim=hidden_state_dim,
                message_dim=message_dim,
                output_transform=link_output_transform,
            ),
            X,
            state,
        )
        hk.set_state("inner_cell_state", state)
        return y

    return forward


class StackedRNNCell(hk.Module):
    def __init__(
        self,
        celltype: str,
        hidden_state_dim,
        stacks: int,
        layernorm: bool = False,
        layernorm_trainable: bool = True,
        name: str | None = None,
    ):
        super().__init__(name)
        cell = {"gru": hk.GRU, "lstm": LSTM}[celltype]

        self.cells = [cell(hidden_state_dim) for _ in range(stacks)]
        self.layernorm = layernorm
        self.layernorm_trainable = layernorm_trainable

    def __call__(self, x, state):
        output = x
        next_state = []
        for i in range(len(self.cells)):
            output, next_state_i = self.cells[i](output, state[i])
            next_state.append(next_state_i)

            if self.layernorm:
                output = hk.LayerNorm(
                    -1, self.layernorm_trainable, self.layernorm_trainable
                )(output)

        return output, jnp.stack(next_state)


class LSTM(hk.RNNCore):
    def __init__(self, hidden_size: int, name=None):
        super().__init__(name=name)
        self.hidden_size = hidden_size

    def __call__(
        self,
        inputs: jax.Array,
        prev_state: jax.Array,
    ):
        if len(inputs.shape) > 2 or not inputs.shape:
            raise ValueError(f"LSTM input must be rank-1 or rank-2; not {inputs.shape}")
        prev_state_h = prev_state[: self.hidden_size]
        prev_state_c = prev_state[self.hidden_size :]
        x_and_h = jnp.concatenate([inputs, prev_state_h], axis=-1)
        gated = hk.Linear(4 * self.hidden_size)(x_and_h)
        i, g, f, o = jnp.split(gated, indices_or_sections=4, axis=-1)
        f = jax.nn.sigmoid(f + 1)  # Forget bias, as in sonnet.
        c = f * prev_state_c + jax.nn.sigmoid(i) * jnp.tanh(g)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)
        return h, jnp.concatenate((h, c))

    def initial_state(self, batch_size: int | None):
        raise NotImplementedError


class RING(ml_base.AbstractFilter):
    def __init__(
        self,
        params=None,
        lam=None,
        jit: bool = True,
        name=None,
        forward_factory=make_ring,
        **kwargs,
    ):
        "Untrained RING network"
        self.forward_lam_factory = partial(forward_factory, **kwargs)
        self.params = self._load_params(params)
        self.lam = lam
        self._name = name

        if jit:
            self.apply = jax.jit(self.apply, static_argnames="lam")

    def apply(self, X, params=None, state=None, y=None, lam=None):
        if lam is None:
            assert self.lam is not None
            lam = self.lam

        return super().apply(X, params, state, y, tuple(lam))

    def init(self, bs: Optional[int] = None, X=None, lam=None, seed: int = 1):
        assert X is not None, "Providing `X` via in `ringnet.init(X=X)` is required"
        if bs is not None:
            assert X.ndim == 4

        if X.ndim == 4:
            if bs is not None:
                assert bs == X.shape[0]
            else:
                bs = X.shape[0]
            X = X[0]

        # (T, N, F) -> (1, N, F) for faster .init call
        X = X[0:1]

        if lam is None:
            assert self.lam is not None
            lam = self.lam

        key = jax.random.PRNGKey(seed)
        params, state = self.forward_lam_factory(lam=lam).init(key, X)

        if bs is not None:
            state = jax.tree_map(lambda arr: jnp.repeat(arr[None], bs, axis=0), state)

        return params, state

    def _apply_batched(self, X, params, state, y, lam):
        if (params is None and self.params is None) or state is None:
            _params, _state = self.init(bs=X.shape[0], X=X, lam=lam)

        if params is None and self.params is None:
            params = _params
        elif params is None:
            params = self.params
        else:
            pass

        if state is None:
            state = _state

        yhat, next_state = jax.vmap(
            self.forward_lam_factory(lam=lam).apply, in_axes=(None, 0, 0)
        )(params, state, X)

        return yhat, next_state

    @staticmethod
    def _load_params(params: str | dict | None | Path):
        assert isinstance(params, (str, dict, type(None), Path))
        if isinstance(params, (Path, str)):
            return pickle_load(params)
        return params

    def nojit(self) -> "RING":
        ringnet = RING(params=self.params, lam=self.lam, jit=False)
        ringnet.forward_lam_factory = self.forward_lam_factory
        return ringnet

    def _pre_save(self, params=None, lam=None) -> None:
        if params is not None:
            self.params = params
        if lam is not None:
            self.lam = lam

    @staticmethod
    def _post_load(ringnet: "RING", jit: bool = True) -> "RING":
        if jit:
            ringnet.apply = jax.jit(ringnet.apply, static_argnames="lam")
        return ringnet
