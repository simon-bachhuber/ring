import jax
import numpy as np

import ring


def test_quickstart_exampe():
    T: int = 30  # sequence length     [s]
    Ts: float = 0.01  # sampling interval   [s]
    B: int = 1  # batch size
    lam: list[int] = [0, 1, 2]  # parent array
    N: int = len(lam)  # number of bodies
    T_i: int = int(T / Ts)  # number of timesteps

    X = np.zeros((B, T_i, N, 9))

    ringnet = ring.RING(lam, Ts)
    yhat, state = ringnet.apply(X)
    assert yhat.shape == (B, T_i, N, 4)
    assert state["~"]["inner_cell_state"].shape == (B, N, 2, 400)

    _ = jax.jit(ringnet.apply)(X, state=state)
