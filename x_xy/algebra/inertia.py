import jax.numpy as jnp

from x_xy import base


def inertia_mul_motion(it: base.Inertia, m: base.Motion) -> base.Force:
    ang = it.it_3x3 @ m.ang + jnp.cross(it.h, m.vel)
    vel = it.mass * m.vel - jnp.cross(it.h, m.ang)
    return base.Force(ang, vel)
