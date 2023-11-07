from types import SimpleNamespace

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from tree_utils import batch_concat_acme
from tree_utils import tree_shape

from x_xy import maths
from x_xy import System

parallel_scan = jax.lax.associative_scan


def forward(lru_parameters, input_sequence):
    "Forward pass of the LRU layer. Output y and input_sequence are of shape (L, H)."

    # All LRU parameters
    nu_log, theta_log, B_re, B_im, C_re, C_im, D, gamma_log = lru_parameters

    # Materializing the diagonal of Lambda and projections
    Lambda = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
    B_norm = (B_re + 1j * B_im) * jnp.expand_dims(jnp.exp(gamma_log), axis=-1)
    C = C_re + 1j * C_im

    # Running the LRU + output projection
    # For details on parallel scan, check discussion in Smith et al (2022).
    Lambda_elements = jnp.repeat(Lambda[None, ...], input_sequence.shape[0], axis=0)
    Bu_elements = jax.vmap(lambda u: B_norm @ u)(input_sequence)
    elements = (Lambda_elements, Bu_elements)
    _, inner_states = parallel_scan(binary_operator_diag, elements)  # all x_k
    y = jax.vmap(lambda x, u: (C @ x).real + D * u)(inner_states, input_sequence)

    return y


def init_lru_parameters(key, N, H, r_min=0, r_max=1, max_phase=6.28):
    """Initialize parameters of the LRU layer."""
    # np.random.seed(int(jax.random.randint(key, (), 0, 1000000)))

    # N: state dimension, H: model dimension
    # Initialization of Lambda is complex valued distributed uniformly on ring
    # between r_min and r_max, with phase in [0, max_phase].
    u1 = np.random.uniform(size=(N,))
    u2 = np.random.uniform(size=(N,))
    nu_log = np.log(-0.5 * np.log(u1 * (r_max**2 - r_min**2) + r_min**2))
    theta_log = np.log(max_phase * u2)

    # Glorot initialized Input/Output projection matrices
    B_re = np.random.normal(size=(N, H)) / np.sqrt(2 * H)
    B_im = np.random.normal(size=(N, H)) / np.sqrt(2 * H)
    C_re = np.random.normal(size=(H, N)) / np.sqrt(N)
    C_im = np.random.normal(size=(H, N)) / np.sqrt(N)
    D = np.random.normal(size=(H,))

    # Normalization factor
    diag_lambda = np.exp(-np.exp(nu_log) + 1j * np.exp(theta_log))
    gamma_log = np.log(np.sqrt(1 - np.abs(diag_lambda) ** 2))

    return jax.tree_map(
        jnp.asarray, (nu_log, theta_log, B_re, B_im, C_re, C_im, D, gamma_log)
    )


def binary_operator_diag(element_i, element_j):
    # Binary operator for parallel scan of linear recurrence.
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    return a_j * a_i, a_j * bu_i + bu_j


class UnrolledLRU(nn.Module):
    N: int
    H: int
    r_min: float = 0.0
    r_max: float = 1.0
    max_phase: float = 6.28

    @nn.compact
    def __call__(self, input_sequence):
        lru_parameters = self.param(
            "lru_parameters",
            init_lru_parameters,
            self.N,
            self.H,
            self.r_min,
            self.r_max,
            self.max_phase,
        )
        return forward(lru_parameters, input_sequence)


_batch_module = lambda module: nn.vmap(
    module,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None},
    split_rngs={"params": False},
)


class ResidualBlockLRU(nn.Module):
    N: int
    H: int

    @nn.compact
    def __call__(self, input_sequence):
        assert input_sequence.ndim == 3

        # norm
        # x = nn.LayerNorm()(input_sequence)
        x = input_sequence
        # recurrency
        x = _batch_module(UnrolledLRU)(self.N, self.H)(x)
        # glu
        x = nn.Dense(self.H)(x) * nn.sigmoid(nn.Dense(self.H)(x))
        # skip connection
        x = input_sequence + x
        return x


_MAX_OUTPUT_N_LINKS: int = 20


class LRU_Observer(nn.Module):
    sys: System
    output_dim: int
    hidden_state_dim_lru: int
    hidden_state_dim_encoder: int
    hidden_state_dim_decoder: int
    # embeeding dimension - H
    embed_dim: int
    n_residual_blocks: int

    @nn.compact
    def __call__(self, X):  # {name: {gyr: (bs, L, features), ..., }, ...}
        n_links = self.sys.num_links()
        bs, L = tree_shape(X), tree_shape(X, 1)

        X_flat = jnp.stack(
            [
                batch_concat_acme(X[name], 2)[..., None, :]
                for name in self.sys.link_names
            ],
            axis=-2,
        ).reshape((bs * L, n_links, -1))

        carry, _ = nn.RNN(
            nn.GRUCell(features=self.hidden_state_dim_encoder), return_carry=True
        )(X_flat)
        encoder_state = carry.reshape((bs, L, self.hidden_state_dim_encoder))

        # embeed encoded state; (bs, L, H)
        x = nn.Dense(self.embed_dim)(encoder_state)

        for _ in range(self.n_residual_blocks):
            x = ResidualBlockLRU(self.hidden_state_dim_lru, self.embed_dim)(x)

        # decoder; (bs, L, hidden_state_decoder)
        decoder_state0 = nn.Dense(self.hidden_state_dim_decoder)(x)
        pseudo_input = jnp.repeat(
            jax.nn.one_hot(jnp.arange(n_links), _MAX_OUTPUT_N_LINKS)[None],
            bs * L,
            axis=0,
        )
        # (bs * L, n_links, hidden_state_decoder)
        decoder_state_seq = nn.RNN(nn.GRUCell(features=self.hidden_state_dim_decoder))(
            pseudo_input,
            initial_carry=decoder_state0.reshape(
                (bs * L, self.hidden_state_dim_decoder)
            ),
        )
        # (bs, L, n_links, hidden_state_decoder)
        decoder_state_seq_4d = decoder_state_seq.reshape(
            (bs, L, n_links, self.hidden_state_dim_decoder)
        )

        # create final output
        output = maths.safe_normalize(nn.Dense(self.output_dim)(decoder_state_seq_4d))

        return {
            name: output[..., i, :]
            for i, name in enumerate(self.sys.link_names)
            if self.sys.link_parents[i] != -1
        }


def make_lru_observer(
    sys: System,
    hidden_state_dim_lru: int = 192,
    hidden_state_dim_encoder: int = 96,
    hidden_state_dim_decoder: int = 96,
    embed_dim: int = 192,
    n_residual_blocks: int = 2,
):
    assert sys.num_links() < _MAX_OUTPUT_N_LINKS

    dummy_state = jnp.zeros((1,))
    lru_observer = LRU_Observer(
        sys=sys,
        output_dim=4,
        hidden_state_dim_lru=hidden_state_dim_lru,
        hidden_state_dim_encoder=hidden_state_dim_encoder,
        hidden_state_dim_decoder=hidden_state_dim_decoder,
        embed_dim=embed_dim,
        n_residual_blocks=n_residual_blocks,
    )

    def init(key, X):
        params = lru_observer.init(key, X)
        return params, dummy_state

    def apply(params, state, X):
        yhat = lru_observer.apply(params, X)
        return yhat, state

    return SimpleNamespace(init=init, apply=apply)
