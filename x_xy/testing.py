from typing import Tuple

import jax
import jax.numpy as jnp

from x_xy import base, maths


def basic_coordinate_logic():
    xaxis = jnp.array([1.0, 0, 0])
    yaxis = jnp.array([0, 1.0, 0])
    zaxis = jnp.array([0, 0, 1.0])
    #   A  x      y  B
    #   y--|      |--z
    #
    # E y
    #   |--x

    tA = base.Transform.create(rot=maths.quat_rot_axis(zaxis, -jnp.pi / 2), pos=yaxis)
    tB = base.Transform.create(
        rot=maths.quat_rot_axis(yaxis, -jnp.pi / 2), pos=xaxis + yaxis
    )
    return xaxis, yaxis, zaxis, tA, tB


def example_system() -> Tuple[base.System, dict[int, jax.Array]]:
    """
                         (1)|   (3)|
    (-1) -> Float Base ->(0)|---(2)|---(4)
    """
    xaxis = jnp.array([1.0, 0, 0])

    links = [
        # Floating Base
        base.Link(base.Transform.zero()),
        base.Link(base.Transform.zero()),
        base.Link(base.Transform.create(pos=xaxis)),
        base.Link(base.Transform.create(pos=xaxis)),
        base.Link(base.Transform.create(pos=xaxis)),
    ]
    links = links[0].batch(*links[1:])

    parents = jnp.array([-1, 0, 0, 2, 2])

    joint_types = ["free", "rz", "rz", "rz", "ry"]

    sys = base.System(parents, links, joint_types, 0.01, False)

    q = {
        0: jnp.array([0, 0, 0, 1, 1, 1.0]),
        1: -jnp.pi / 2,
        2: -jnp.pi / 2,
        3: -jnp.pi / 4,
        4: -jnp.pi / 2,
    }

    return sys, q
