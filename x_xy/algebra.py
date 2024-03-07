import jax.numpy as jnp
from ring import base
from ring import maths
from ring import spatial


def inertia_mul_motion(it: base.Inertia, m: base.Motion) -> base.Force:
    ang = it.it_3x3 @ m.ang + jnp.cross(it.h, m.vel)
    vel = it.mass * m.vel - jnp.cross(it.h, m.ang)
    return base.Force(ang, vel)


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


def transform_mul(t2: base.Transform, t1: base.Transform) -> base.Transform:
    """Chains two transformations `t1` and `t2`.
    t1: Plücker A -> Plücker B,
    t2: Plücker B -> Plücker C
    =>
    Returns: Plücker A -> Plücker C
    """
    pos = t1.pos + maths.rotate(t2.pos, maths.quat_inv(t1.rot))
    rot = maths.quat_mul(t2.rot, t1.rot)
    return base.Transform(pos, rot)


def transform_inv(t: base.Transform) -> base.Transform:
    "Inverts the transform. A -> B becomes B -> A"
    pos = maths.rotate(-t.pos, t.rot)
    rot = maths.quat_inv(t.rot)
    return base.Transform(pos, rot)


def transform_move_into_frame(
    t: base.Transform, new_frame: base.Transform
) -> base.Transform:
    """Express transform `t`: A -> B, in frame C using `new_frame`: A -> C.

    Suppose you are given a transform `t` that maps from A -> B.
    Then, you think of this operation as something abstract.
    Then, you want to do this operation that maps from A -> B but
    apply it in frame C. The connection between A -> C is given by `new_frame`.
    """
    q_C_to_A = new_frame.rot
    rot = maths.rotate_quat(t.rot, q_C_to_A)
    pos = maths.rotate(t.pos, q_C_to_A)
    return base.Transform(pos, rot)


def transform_motion(t: base.Transform, m: base.Motion) -> base.Motion:
    """Transforms motion vector `m`.
    t: Plücker A -> Plücker B,
    m: Plücker A
    =>
    Returns: m in Plücker B
    """
    ang = maths.rotate(m.ang, t.rot)
    vel = maths.rotate(-jnp.cross(t.pos, m.ang) + m.vel, t.rot)
    return base.Motion(ang, vel)


def transform_force(t: base.Transform, f: base.Force) -> base.Force:
    """Transforms force vector `f`.
    t: Plücker A -> Plücker B,
    f: Plücker A
    =>
    Returns: f in Plücker B
    """
    ang = maths.rotate(f.ang - jnp.cross(t.pos, f.vel), t.rot)
    vel = maths.rotate(f.vel, t.rot)
    return base.Force(ang, vel)


def transform_inertia(t: base.Transform, it: base.Inertia) -> base.Inertia:
    "Transforms inertia matrix `it`"
    r = t.pos
    I_ = it.it_3x3
    rcross = spatial.cross(r)

    hmr = it.h - it.mass * t.pos
    new_h = maths.rotate(hmr, t.rot)
    new_it_3x3 = maths.rotate_matrix(
        I_ + rcross @ spatial.cross(it.h) + spatial.cross(hmr) @ rcross,
        t.rot,
    )
    return base.Inertia(new_it_3x3, new_h, it.mass)
