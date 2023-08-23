import os

import jax
import jax.numpy as jnp

import x_xy
from x_xy import render
from x_xy.algorithms import dynamics
from x_xy.base import State
from x_xy.io.xml.from_xml import load_sys_from_str


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


def test_shapes():
    sys_str = """
<x_xy model="shape_test">
    <options gravity="0 0 9.81" dt="0.01" />
    <worldbody>
        <geom type="sphere" mass="1" pos="0 0 0" dim="0.3" color="white" />
        <geom type="box" mass="1" pos="-1 0 0" quat="1 0 1 0" dim="1 0.3 0.2" color="0.8 0.3 1 0" />
        <geom type="cylinder" mass="1" pos="1 0 0.5" quat="0.75 0 0 0.25" dim="0.3 1" color="0.2 0.8 0.5" />
        <geom type="capsule" mass="1" pos="0 0 -1" dim="0.3 2" />

        <body name="dummy" pos="0 0 0" quat="1 0 0 0" joint="ry" />
    </worldbody>
</x_xy>
    """

    sys = load_sys_from_str(sys_str)

    state = State.create(sys, q=None)

    step_fn = jax.jit(dynamics.step)

    state = step_fn(sys, state, jnp.zeros_like(state.qd))

    x = state.x.batch()

    render.animate("figures/example.png", sys, x, fmt="png")


if __name__ == "__main__":
    test_animate(pytest=False)
