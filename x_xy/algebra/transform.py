import jax.numpy as jnp

from .. import base
from .. import maths


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
    rcross = maths.spatial.cross(r)

    hmr = it.h - it.mass * t.pos
    new_h = maths.rotate(hmr, t.rot)
    new_it_3x3 = maths.rotate_matrix(
        I_ + rcross @ maths.spatial.cross(it.h) + maths.spatial.cross(hmr) @ rcross,
        t.rot,
    )
    return base.Inertia(new_it_3x3, new_h, it.mass)
