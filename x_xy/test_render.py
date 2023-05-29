import os

import jax
import jax.numpy as jnp

import x_xy
from x_xy import render
from x_xy.algorithms import dynamics


def test_animate(pytest=True):
    dt = 1e-2
    filename = "animation"
    xml = "test_double_pendulum"
    sys = x_xy.io.load_example(xml)
    sys = sys.replace(dt=dt)

    q = jnp.array([0, 1.0])
    qd = jnp.zeros((sys.qd_size(),))

    state = x_xy.base.State.create(sys, q, qd)

    T = 10
    if pytest:
        T = 1

    step_fn = jax.jit(dynamics.step)

    xs = []
    for _ in range(int(T / sys.dt)):
        state = step_fn(sys, state, jnp.zeros_like(state.qd))
        xs.append(state.x)

    xs = xs[0].batch(*xs[1:])

    fmts = ["mp4"]
    if pytest:
        fmts += ["gif"]

    for fmt in fmts:
        render.animate(filename, sys, xs, fmt=fmt)
        if pytest:
            os.system(f"rm animation.{fmt}")


if __name__ == "__main__":
    test_animate(pytest=False)
