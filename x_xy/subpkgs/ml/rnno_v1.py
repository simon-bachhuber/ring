import haiku as hk
import jax
import jax.numpy as jnp
from tree_utils import batch_concat_acme

from x_xy.maths import safe_normalize

from .ml_utils import InitApplyFnPair

complex = dict(rnn_layers=(400, 300), linear_layers=(200, 100, 50, 50, 25, 25))
medium = dict(rnn_layers=(300, 200), linear_layers=(100, 50, 50, 25, 25))
shallow = dict(rnn_layers=(100,), linear_layers=(25,))
tiny = dict(rnn_layers=(10,), linear_layers=())
complexities = dict(complex=complex, medium=medium, shallow=shallow, tiny=tiny)


def make_rnno_v1(
    lam: list[int],
    rnn_layers=(100,),
    linear_layers=(),
    keep_toRoot_output: bool = False,
    layernorm=True,
    act_fn_linear=jax.nn.relu,
    act_fn_rnn=jax.nn.elu,
) -> InitApplyFnPair:
    """RNN-neural net.
    (bs, time, features)
    """

    bodies = [i for i, p in enumerate(lam) if p != -1 or keep_toRoot_output]
    N = len(bodies)

    @hk.without_apply_rng
    @hk.transform_with_state
    def forward_fn(X):
        X = batch_concat_acme(X, num_batch_dims=2)
        bs = X.shape[0]

        for i, n_units in enumerate(rnn_layers):

            state = hk.get_state(f"rnn_{i}", shape=[bs, n_units], init=jnp.zeros)
            X, state = hk.dynamic_unroll(hk.GRU(n_units), X, state, time_major=False)
            hk.set_state(f"rnn_{i}", state)

            # layer-norm
            if layernorm:
                X = hk.LayerNorm(axis=-1, create_scale=False, create_offset=False)(X)

            X = act_fn_rnn(X)

        for n_units in linear_layers:
            X = hk.Linear(n_units)(X)

            X = act_fn_linear(X)

        out_dim = N * 4
        X = hk.Linear(out_dim)(X)

        X_dict = dict()
        start = 0
        for i in bodies:
            X_dict[i] = jax.vmap(jax.vmap(safe_normalize))(X[:, :, start : (start + 4)])
            start += 4
        return X_dict

    def init(key, *args):
        params, state = forward_fn.init(key, *args)
        return params, jax.tree_map(lambda arr: arr[0], state)

    return InitApplyFnPair(init=init, apply=forward_fn.apply)
