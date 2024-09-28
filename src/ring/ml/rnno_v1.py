from typing import Optional, Sequence

import haiku as hk
import jax
import jax.numpy as jnp

from .ringnet import LSTM


def rnno_v1_forward_factory(
    output_dim: int,
    rnn_layers: Sequence[int] = (400, 300),
    linear_layers: Sequence[int] = (200, 100, 50, 50, 25, 25),
    layernorm: bool = True,
    act_fn_linear=jax.nn.relu,
    act_fn_rnn=jax.nn.elu,
    lam: Optional[tuple[int]] = None,
    celltype: str = "gru",
):
    # unused
    del lam

    if celltype == "gru":
        _cell = hk.GRU
        _factor = 1
    elif celltype == "lstm":
        _cell = LSTM
        _factor = 2
    else:
        raise NotImplementedError

    @hk.without_apply_rng
    @hk.transform_with_state
    def forward_fn(X):
        assert X.shape[-2] == 1
        X = X[..., 0, :]

        for i, n_units in enumerate(rnn_layers):
            state = hk.get_state(f"rnn_{i}", shape=[n_units * _factor], init=jnp.zeros)
            X, state = hk.dynamic_unroll(_cell(n_units), X, state)
            hk.set_state(f"rnn_{i}", state)

            if layernorm:
                X = hk.LayerNorm(axis=-1, create_scale=False, create_offset=False)(X)
            X = act_fn_rnn(X)

        for n_units in linear_layers:
            X = hk.Linear(n_units)(X)
            X = act_fn_linear(X)

        y = hk.Linear(output_dim)(X)
        return y[..., None, :]

    return forward_fn
