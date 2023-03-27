import os

import jax.numpy as jnp

from x_xy.base import (
    Box,
    GeometryCollection,
    Joint,
    JointType,
    Link,
    State,
    System,
    Transform,
)
from x_xy.dynamics import simulation_step
from x_xy.render import Launcher, Renderer, animate

box = Box(
    10.0,
    jnp.array([0.5, 0, 0]),
    1.0,
    0.25,
    0.2,
    vispy_kwargs=dict(edge_color="black", color="white"),
)

geom_coll = GeometryCollection([[box], [box], [box]])


def make_system():

    l1 = Link.create(
        Transform.create(pos=jnp.array([1.0, 1.0, 1])),
        Joint(JointType.RevoluteY, damping=1),
        box,
    )
    l2 = Link.create(
        Transform.create(pos=jnp.array([1.0, 0.0, 0])),
        Joint(JointType.RevoluteY, damping=1),
        box,
    )
    l3 = Link.create(
        Transform.create(pos=jnp.array([1.0, 0.0, 0])),
        Joint(JointType.RevoluteY, damping=1),
        box,
    )
    sys = System(jnp.array([-1, 0, 1]), l1.batch(l2, l3))
    q = jnp.array(([0.0, jnp.pi, 0.0]))
    qd = jnp.zeros((sys.N,))
    state = State(q, qd, Transform.zero((sys.N,)), Transform.zero((sys.N,)))
    return sys, state


T = 1.0
timestep = 0.01


def simulate():
    sys, state = make_system()
    taus = jnp.zeros((sys.N,))
    transforms = []

    for _ in range(int(T // timestep)):
        transforms.append(state.x)
        state = simulation_step(sys, state, taus, timestep=timestep)

    stacked_transforms = transforms[0].batch(*transforms[1:])
    return stacked_transforms


transforms = simulate()


def _test_launcher():
    renderer = Renderer(geom_coll)
    launcher = Launcher(renderer, transforms.pos, transforms.rot, timestep)
    launcher.start()
    launcher.reset()


def test_animate():
    renderer = Renderer(geom_coll, headless=True)
    animate("animation.gif", renderer, transforms.pos, transforms.rot, timestep)
    os.system("rm animation.gif")
