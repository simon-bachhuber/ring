import jax.numpy as jnp

from .. import base


def motion_dot(m: base.Motion, f: base.Force) -> base.Scalar:
    return m.ang @ f.ang + m.vel @ f.vel


def motion_cross(m1: base.Motion, m2: base.Motion) -> base.Motion:
    ang = jnp.cross(m1.ang, m2.ang)
    vel = jnp.cross(m1.ang, m2.vel) + jnp.cross(m1.vel, m2.ang)
    return base.Motion(ang, vel)


def motion_cross_star(m: base.Motion, f: base.Force) -> base.Force:
    ang = jnp.cross(m.ang, f.ang) + jnp.cross(m.vel, f.vel)
    vel = jnp.cross(m.ang, f.vel)
    return base.Force(ang, vel)
