from types import SimpleNamespace
from typing import Callable, Optional

import haiku as hk
import jax
import jax.numpy as jnp
from jax.random import normal
from jax.random import uniform
import tree_utils

from x_xy import base
from x_xy import scan_sys
from x_xy.maths import safe_normalize
from x_xy.subpkgs import ml


class RNNOFilter:
    def __init__(
        self,
        identifier: Optional[str] = None,
        params: Optional[dict] = None,
        key: jax.Array = jax.random.PRNGKey(1),
        **rnno_kwargs,
    ):
        self._identifier = identifier
        self.key = key
        self.params = params
        self.rnno_fn = lambda sys: ml.make_rnno(sys, **rnno_kwargs)

    def init(self, sys, X_t0):
        X_batched = tree_utils.to_3d_if_2d(tree_utils.add_batch_dim(X_t0), strict=True)
        self.rnno = self.rnno_fn(sys)
        params, self.state = self.rnno.init(self.key, X_batched)
        if self.params is None:
            self.params = params

    def predict(self, X: dict) -> dict:
        assert tree_utils.tree_ndim(X) == 3
        bs = tree_utils.tree_shape(X)
        state = jax.tree_map(lambda arr: jnp.repeat(arr[None], bs, axis=0), self.state)
        return self.rnno.apply(self.params, state, X)[0]

    def identifier(self) -> str:
        if self._identifier is None:
            raise RuntimeError("No `identifier` was given.")
        return self._identifier


def _tree(sys, f):
    return scan_sys(
        sys,
        f,
        "lll",
        list(range(sys.num_links())),
        sys.link_parents,
        sys.link_names,
    )


def _make_rnno_cell_apply_fn(
    sys,
    inner_cell,
    send_msg,
    send_output,
    hidden_state_dim,
    message_dim,
    send_message_stop_grads,
    output_transform: Callable,
):
    parent_array = jnp.array(sys.link_parents, dtype=jnp.int32)

    def _rnno_cell_apply_fn(inputs, prev_state):
        empty_message = jnp.zeros((1, message_dim))
        mailbox = jnp.repeat(empty_message, sys.num_links(), axis=0)

        # message is sent using the hidden state of the last cell
        # for LSTM `prev_state` is of shape (2 * hidden_state_dim) du to cell state
        prev_last_hidden_state = prev_state[:, -1, :hidden_state_dim]

        # lru cell has complex valued hidden state
        if prev_last_hidden_state.dtype == jnp.complex64:
            prev_last_hidden_state = prev_last_hidden_state.real

        if send_message_stop_grads:
            prev_last_hidden_state = jax.lax.stop_gradient(prev_last_hidden_state)
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

        mailbox = jax.vmap(accumulate_message)(jnp.arange(sys.num_links()))

        def cell_input(_, __, i: int, p: int, name: str):
            assert name in inputs, f"name: {name} not found in {list(inputs.keys())}"
            local_cell_input = tree_utils.batch_concat_acme(
                (inputs[name], msg[p], mailbox[i]), num_batch_dims=0
            )
            return local_cell_input

        stacked_cell_input = _tree(sys, cell_input)

        def update_state(cell_input, state):
            cell_output, state = inner_cell(cell_input, state)
            output = output_transform(send_output(cell_output))
            return output, state

        y, state = jax.vmap(update_state)(stacked_cell_input, prev_state)

        outputs = {
            sys.idx_to_name(i): y[i]
            for i in range(sys.num_links())
            if sys.link_parents[i] != -1
        }
        return outputs, state

    return _rnno_cell_apply_fn


def make_rnno(
    sys: base.System,
    hidden_state_dim: int = 400,
    message_dim: int = 200,
    cell_type: str = "gru",
    stack_rnn_cells: int = 1,
    send_message_n_layers: int = 1,
    send_message_method: str = "mlp",
    send_message_init: hk.initializers.Initializer = hk.initializers.Orthogonal(),
    send_message_stop_grads: bool = False,
    link_output_dim: int = 4,
    link_output_normalize: bool = True,
    link_output_transform: Optional[Callable] = None,
    layernorm: bool = False,
) -> SimpleNamespace:
    "Expects batched inputs."

    if cell_type == "gru":
        cell = hk.GRU
        hidden_state_dtype = jnp.float32
        hidden_state_init = hidden_state_dim
    elif cell_type == "lstm":
        cell = LSTM
        hidden_state_init = hidden_state_dim * 2
        hidden_state_dtype = jnp.float32
    elif cell_type == "lru":
        cell = LRU
        hidden_state_init = hidden_state_dim
        hidden_state_dtype = jnp.complex64
    else:
        raise NotImplementedError

    if link_output_normalize:
        assert link_output_transform is None
        link_output_transform = safe_normalize

    @hk.without_apply_rng
    @hk.transform_with_state
    def forward(X):
        if send_message_method == "mlp":
            send_msg = hk.nets.MLP(
                [hidden_state_dim] * send_message_n_layers + [message_dim]
            )
        elif send_message_method == "matrix":
            matrix = hk.get_state(
                "send_msg_matrix",
                [message_dim, hidden_state_dim],
                init=send_message_init,
            )
            send_msg = lambda hidden_state: matrix @ hidden_state
        else:
            raise NotImplementedError

        inner_cell = StackedRNNCell(
            cell, hidden_state_dim, stack_rnn_cells, layernorm=layernorm
        )
        send_output = hk.nets.MLP([hidden_state_dim, link_output_dim])
        state = hk.get_state(
            "inner_cell_state",
            [sys.num_links(), stack_rnn_cells, hidden_state_init],
            init=jnp.zeros,
            dtype=hidden_state_dtype,
        )

        y, state = hk.dynamic_unroll(
            _make_rnno_cell_apply_fn(
                sys,
                inner_cell,
                send_msg,
                send_output,
                hidden_state_dim,
                message_dim,
                send_message_stop_grads,
                output_transform=link_output_transform,
            ),
            X,
            state,
        )
        hk.set_state("inner_cell_state", state)
        return y

    def init(key, X):
        "X.shape (bs, timesteps, features)"
        X = tree_utils.to_2d_if_3d(X, strict=True)
        return forward.init(key, X)

    def apply(params, state, X):
        """
        params: (features)
        state.shape (bs, features)
        X.shape (bs, timesteps, features)

        Returns: (yhat, state)
        yhat.shape (bs, timesteps, features)
        state.shape (bs, features)
        """
        assert tree_utils.tree_ndim(X) == 3
        return jax.vmap(forward.apply, in_axes=(None, 0, 0))(params, state, X)

    return SimpleNamespace(init=init, apply=apply)


class StackedRNNCell(hk.Module):
    def __init__(
        self,
        cell,
        hidden_state_dim,
        stacks: int,
        layernorm: bool = False,
        name: str | None = None,
    ):
        super().__init__(name)

        self.cells = []
        if isinstance(cell, LRU):
            self.cells.append(LRU(hidden_state_dim, embed_size=hidden_state_dim))
            stacks -= 1

        self.cells.extend([cell(hidden_state_dim) for _ in range(stacks)])

        self.layernorm = layernorm

    def __call__(self, x, state):
        output = x
        next_state = []
        for i in range(len(self.cells)):
            output, next_state_i = self.cells[i](output, state[i])
            next_state.append(next_state_i)

            if self.layernorm:
                output = hk.LayerNorm(-1, True, True)(output)

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
            raise ValueError("LSTM input must be rank-1 or rank-2.")
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


def _zeros_lru_parameters(N, H) -> tuple:
    zeros = lambda *args: jnp.zeros(tuple(args))
    return (
        zeros(N),
        zeros(N),
        zeros(N, H),
        zeros(N, H),
        zeros(H, N),
        zeros(H, N),
        zeros(H),
        zeros(N),
    )


def _flatten_lru_parameters(lru_params: tuple) -> jax.Array:
    return jax.flatten_util.ravel_pytree(lru_params)[0]


def _unflatten_lru_parameters(lru_params: jax.Array, N, H) -> tuple:
    _, unflatten = jax.flatten_util.ravel_pytree(_zeros_lru_parameters(N, H))
    return unflatten(lru_params)


def _lru_timestep(lru_parameters: jax.Array, inner_state_tm1, u_t, N, H):
    nu_log, theta_log, B_re, B_im, C_re, C_im, D, gamma_log = _unflatten_lru_parameters(
        lru_parameters, N, H
    )

    # Materializing the diagonal of Lambda and projections
    Lambda = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
    B_norm = (B_re + 1j * B_im) * jnp.expand_dims(jnp.exp(gamma_log), axis=-1)
    C = C_re + 1j * C_im

    # Running the LRU + output projection
    inner_state_t = Lambda * inner_state_tm1 + B_norm @ u_t
    y = (C @ inner_state_t).real + D * u_t
    return y, inner_state_t


def _build_init_lru_parameters(N, H):
    def _init_lru_parameters(shape, dtype):
        """Initialize parameters of the LRU layer."""
        r_min = jnp.array(0.0)
        r_max = jnp.array(1.0)
        max_phase = jnp.array(2 * jnp.pi)

        # N: state dimension, H: model dimension
        # Initialization of Lambda is complex valued distributed uniformly on ring
        # between r_min and r_max, with phase in [0, max_phase].
        u1 = uniform(hk.next_rng_key(), shape=(N,))
        u2 = uniform(hk.next_rng_key(), shape=(N,))
        nu_log = jnp.log(-0.5 * jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        theta_log = jnp.log(max_phase * u2)

        # Glorot initialized Ijnput/Output projection matrices
        B_re = normal(hk.next_rng_key(), shape=(N, H)) / jnp.sqrt(2 * H)
        B_im = normal(hk.next_rng_key(), shape=(N, H)) / jnp.sqrt(2 * H)
        C_re = normal(hk.next_rng_key(), shape=(H, N)) / jnp.sqrt(N)
        C_im = normal(hk.next_rng_key(), shape=(H, N)) / jnp.sqrt(N)
        D = normal(hk.next_rng_key(), shape=(H,))

        # Normalization factor
        diag_lambda = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
        gamma_log = jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))

        params = nu_log, theta_log, B_re, B_im, C_re, C_im, D, gamma_log
        return _flatten_lru_parameters(params)

    return _init_lru_parameters


class LRU(hk.RNNCore):
    def __init__(self, hidden_size: int, embed_size: Optional[int] = None, name=None):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.embed_size = embed_size

    def __call__(
        self,
        inputs: jax.Array,
        prev_state: jax.Array,
    ):
        if self.embed_size is not None:
            H = self.embed_size
            hk.Linear(self.embed_size, name="encoder")
        else:
            H = inputs.size

        lru_params_flat_size = _flatten_lru_parameters(
            _zeros_lru_parameters(self.hidden_size, H)
        ).size
        lru_params = hk.get_parameter(
            "lru_parameters",
            shape=[lru_params_flat_size],
            init=_build_init_lru_parameters(self.hidden_size, H),
        )
        y, next_state = _lru_timestep(
            lru_params, prev_state, inputs, self.hidden_size, H
        )

        # glu + skip connection
        y = (hk.Linear(H)(y) * jax.nn.sigmoid(hk.Linear(H)(y))) + inputs

        return y, next_state

    def initial_state(self, batch_size: int | None):
        raise NotImplementedError
