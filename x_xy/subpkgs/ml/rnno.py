from types import SimpleNamespace

import haiku as hk
import jax
import jax.numpy as jnp
import tree_utils

from x_xy import base
from x_xy import scan_sys
from x_xy.maths import safe_normalize


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
    send_quat,
    hidden_state_dim,
    message_dim,
    send_message_stop_grads,
):
    parent_array = jnp.array(sys.link_parents, dtype=jnp.int32)

    def _rnno_cell_apply_fn(inputs, prev_state):
        empty_message = jnp.zeros((1, message_dim))
        mailbox = jnp.repeat(empty_message, sys.num_links(), axis=0)
        # message is sent using the hidden state of the last cell
        # for LSTM `prev_state` is of shape (2 * hidden_state_dim) du to cell state
        prev_last_hidden_state = prev_state[:, -1, :hidden_state_dim]
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
            output, state = inner_cell(cell_input, state)
            return safe_normalize(send_quat(output)), state

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
    use_gru: bool = True,
    stack_rnn_cells: int = 1,
    send_message_n_layers: int = 1,
    send_message_method: str = "mlp",
    send_message_init: hk.initializers.Initializer = hk.initializers.Orthogonal(),
    send_message_stop_grads: bool = False,
) -> SimpleNamespace:
    "Expects unbatched inputs. Batching via `vmap`"

    if use_gru:
        cell = hk.GRU
        hidden_state_init = hidden_state_dim
    else:
        cell = LSTM
        hidden_state_init = hidden_state_dim * 2

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

        inner_cell = StackedRNNCell(cell, hidden_state_dim, stack_rnn_cells)
        send_quat = hk.nets.MLP([hidden_state_dim, 4])
        state = hk.get_state(
            "inner_cell_state",
            [sys.num_links(), stack_rnn_cells, hidden_state_init],
            init=jnp.zeros,
        )

        y, state = hk.dynamic_unroll(
            _make_rnno_cell_apply_fn(
                sys,
                inner_cell,
                send_msg,
                send_quat,
                hidden_state_dim,
                message_dim,
                send_message_stop_grads,
            ),
            X,
            state,
        )
        hk.set_state("inner_cell_state", state)
        return y

    def init(key, X):
        X = tree_utils.to_2d_if_3d(X, strict=True)
        return forward.init(key, X)

    def apply(params, state, X):
        assert tree_utils.tree_ndim(X) == 3
        return jax.vmap(forward.apply, in_axes=(None, 0, 0))(params, state, X)

    return SimpleNamespace(init=init, apply=apply)


class StackedRNNCell(hk.Module):
    def __init__(self, cell, hidden_state_dim, stacks: int, name: str | None = None):
        super().__init__(name)
        self.cells = [cell(hidden_state_dim) for _ in range(stacks)]

    def __call__(self, x, state):
        output = x
        next_state = []
        for i in range(len(self.cells)):
            output, next_state_i = self.cells[i](output, state[i])
            next_state.append(next_state_i)
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
