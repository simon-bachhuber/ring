import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np
import tree_utils as tu

from ring import algebra
from ring import base
from ring import maths
from ring import spatial
from ring.base import Force
from ring.base import Inertia
from ring.base import Motion
from ring.base import Transform

"""Tests that compare directly to matrix implementations of the same operations."""

r = jnp.array([2.7, 1.1, 0.2])
angle = jnp.array(0.67)
E1 = spatial.rx(angle)
E2 = spatial.rx(angle) @ spatial.ry(angle) @ spatial.rz(angle)


def test_mat_transform():
    for E in [E1, E2]:
        X = spatial.X_transform(E, r)
        t = Transform(r, maths.quat_from_3x3(E))
        assert jnp.allclose(X, t.as_matrix())
        assert jnp.allclose(X @ X, algebra.transform_mul(t, t).as_matrix(), atol=1e-6)
        assert jnp.allclose(
            jnp.linalg.inv(X), algebra.transform_inv(t).as_matrix(), atol=1e-6
        )


def toy_I_3x3(seed):
    np.random.seed(seed)
    I_ = np.random.exponential(size=(3, 3))
    I_3x3 = I_ + I_.T
    return I_3x3


def test_mat_inertia():
    for i in range(3):  # arbitrary number
        I_3x3 = toy_I_3x3(i)
        mass = 10.0
        I_mat = spatial.mcI(mass, r, I_3x3)
        inertia = Inertia.create(mass, Transform.create(pos=r), I_3x3)
        assert jnp.allclose(I_mat, inertia.as_matrix())
        for E in [E1, E2]:
            X = spatial.X_transform(E, r)
            X_inv = jnp.linalg.inv(X)
            X_star = X_inv.T
            t = Transform(r, maths.quat_from_3x3(E))
            assert jnp.allclose(
                algebra.transform_inertia(t, inertia).as_matrix(),
                X_star @ I_mat @ X_inv,
                atol=1e-5,
            )
            assert jnp.allclose(
                algebra.transform_inertia(
                    algebra.transform_inv(t), inertia
                ).as_matrix(),
                X.T @ I_mat @ X,
                atol=1e-5,
            )


"""Tests are all taken from Page 247 of "Rigid Body Dynamics Algorithms" Book.
"""


class PRNGThread:
    def __init__(self, key: jrand.PRNGKey):
        self.key = key

    def next(self, num: int = 1):
        keys = jrand.split(self.key, num + 1)
        self.key = keys[0]
        return keys[1]


def inertia_3x3_for_box(mass, dim_x, dim_y, dim_z):
    inertia = (
        1
        / 12
        * mass
        * jnp.array(
            [
                [dim_y**2 + dim_z**2, 0, 0],
                [0, dim_x**2 + dim_z**2, 0],
                [0, 0, dim_x**2 + dim_y**2],
            ]
        )
    )
    return inertia


jax_key = jrand.PRNGKey(1)
key = PRNGThread(jax_key)

m_ang1, m_ang2, f_ang1, f_ang2 = jrand.uniform(key.next(), shape=(4, 3), minval=-1)
m_vel1, m_vel2, f_vel1, f_vel2 = jrand.uniform(key.next(), shape=(4, 3), minval=-1)
m1, m2 = Motion(m_ang1, m_vel1), Motion(m_ang2, m_vel2)
f1, f2 = Force(f_ang1, f_vel1), Force(f_ang2, f_vel2)
rot1, rot2 = maths.quat_random(key.next(), batch_shape=(2,))
pos1, pos2 = jrand.uniform(key.next(), shape=(2, 3), minval=-1)
t1 = Transform(pos1, rot1)
t2 = Transform(pos2, rot2)

it_3x3_1 = inertia_3x3_for_box(10, 0.3, 0.3, 0.5)
it_3x3_2 = jnp.array([[1.0, 0.2, 0.5], [0.2, 2.0, 0.4], [0.5, 0.4, 1.0]])
com1, com2 = jrand.uniform(key.next(), shape=(2, 3), minval=-1)
mass1, mass2 = jrand.uniform(key.next(), shape=(2,), minval=1e-12)
it1 = Inertia.create(mass1, Transform.create(pos=com1), it_3x3_1)
it2 = Inertia.create(mass2, Transform.create(pos=com2), it_3x3_2)


def test_flatten():
    assert jnp.allclose(m1.flatten(), jnp.concatenate((m1.ang, m1.vel)))
    assert jnp.allclose(f1.flatten(), jnp.concatenate((f1.ang, f1.vel)))


def test_concat():
    assert tu.tree_close(Motion.zero((2,)), Motion.zero().batch(Motion.zero()))


def test_transform_create():
    t = Transform.create(pos=jnp.array([1.0, 1, 1]))
    assert tu.tree_close(
        t, Transform(jnp.array([1.0, 1.0, 1.0]), jnp.array([1.0, 0, 0, 0]))
    )

    t = Transform.create(rot=jnp.array([1.0, 0, 0, 0]))
    assert tu.tree_close(
        t, Transform(jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 0, 0, 0]))
    )


def test_transform_transform():
    rot = maths.quat_mul(rot1, rot2)
    pos = pos2 + maths.rotate(pos1, maths.quat_inv(rot2))
    t = Transform(pos, rot)
    that = algebra.transform_mul(t1, t2)
    assert tu.tree_close(t, that)
    assert jnp.allclose(t.as_matrix(), that.as_matrix())


def test_transform_inv():
    for t in [t1, t2]:
        t_inv = algebra.transform_inv(t)
        assert jnp.allclose(
            algebra.transform_mul(t, t_inv).as_matrix(), jnp.eye(6), atol=1e-7
        )
        # TODO
        assert jnp.allclose(
            algebra.transform_mul(t_inv, t).as_matrix(), jnp.eye(6), atol=1e-6
        )
        assert tu.tree_close(
            algebra.transform_mul(t, t_inv), Transform.zero(), atol=1e-7
        )


def test_transform_motion():
    for m, t in zip([m1, m2], [t1, t2]):
        assert tu.tree_close(
            algebra.transform_motion(t, m),
            Motion(
                maths.rotate(m.ang, t.rot),
                maths.rotate(m.vel - jnp.cross(t.pos, m.ang), t.rot),
            ),
        )


def test_transform_inv_motion():
    for m, t in zip([m1, m2], [t1, t2]):
        inv = lambda vec: maths.rotate(vec, maths.quat_inv(t.rot))
        assert tu.tree_close(
            algebra.transform_motion(algebra.transform_inv(t), m),
            Motion(inv(m.ang), inv(m.vel) + jnp.cross(t.pos, inv(m.ang))),
        )


def test_transform_force():
    for f, t in zip([f1, f2], [t1, t2]):
        assert tu.tree_close(
            algebra.transform_force(t, f),
            Force(
                maths.rotate(f.ang - jnp.cross(t.pos, f.vel), t.rot),
                maths.rotate(f.vel, t.rot),
            ),
        )


def test_transform_inv_force():
    for f, t in zip([f1, f2], [t1, t2]):
        inv = lambda vec: maths.rotate(vec, maths.quat_inv(t.rot))
        assert tu.tree_close(
            algebra.transform_force(algebra.transform_inv(t), f),
            Force(inv(f.ang) + jnp.cross(t.pos, inv(f.vel)), inv(f.vel)),
        )


def test_tranform_inv_inertia():
    # X^T @ I @ X
    for it, t in zip([it1, it2], [t1, t2]):
        inv = lambda vec: maths.rotate(vec, maths.quat_inv(t.rot))
        new_h = inv(it.h) + it.mass * t.pos
        rcross = spatial.cross(t.pos)
        it_3x3 = (
            maths.rotate_matrix(it.it_3x3, maths.quat_inv(t.rot))
            - rcross @ spatial.cross(inv(it.h))
            - spatial.cross(new_h) @ rcross
        )
        assert tu.tree_close(
            algebra.transform_inertia(algebra.transform_inv(t), it),
            Inertia(it_3x3, new_h, it.mass),
        )


def test_motion_cross_motion():
    for ma in [m1, m2]:
        for mb in [m1, m2]:
            assert jnp.allclose(
                algebra.motion_cross(ma, mb).as_matrix(),
                jnp.concatenate(
                    (
                        jnp.cross(ma.ang, mb.ang),
                        jnp.cross(ma.ang, mb.vel) + jnp.cross(ma.vel, mb.ang),
                    )
                ),
            )


def test_motion_cross_force():
    for m in [m1, m2]:
        for f in [f1, f2]:
            assert jnp.allclose(
                algebra.motion_cross_star(m, f).as_matrix(),
                jnp.concatenate(
                    (
                        jnp.cross(m.ang, f.ang) + jnp.cross(m.vel, f.vel),
                        jnp.cross(m.ang, f.vel),
                    )
                ),
            )


def test_inertia_dot_motion():
    for m in [m1, m2]:
        for it in [it1, it2]:
            assert tu.tree_close(
                algebra.inertia_mul_motion(it, m),
                Force(
                    it.it_3x3 @ m.ang + jnp.cross(it.h, m.vel),
                    it.mass * m.vel - jnp.cross(it.h, m.ang),
                ),
            )


def test_motion_dot_force():
    assert jnp.isclose(algebra.motion_dot(m1, f1), m1.flatten() @ f1.flatten())


"""Tests from 15.04.23 onwards"""


def basic_coordinate_logic():
    xaxis = jnp.array([1.0, 0, 0])
    yaxis = jnp.array([0, 1.0, 0])
    zaxis = jnp.array([0, 0, 1.0])
    #   A  x      y  B
    #   y--|      |--z
    #
    # E y
    #   |--x

    tA = base.Transform.create(rot=maths.quat_rot_axis(zaxis, jnp.pi / 2), pos=yaxis)
    tB = base.Transform.create(
        rot=maths.quat_rot_axis(yaxis, jnp.pi / 2), pos=xaxis + yaxis
    )
    return xaxis, yaxis, zaxis, tA, tB


def test_transform_move_into_frame():
    xaxis, yaxis, zaxis, tA, tB = basic_coordinate_logic()

    t_A_B = algebra.transform_mul(tB, algebra.transform_inv(tA))

    # first check that the position vector is ok
    assert tu.tree_close(t_A_B.pos, -yaxis, atol=1e-7)

    t_A_B_in_B = algebra.transform_move_into_frame(t_A_B, t_A_B)
    assert tu.tree_close(t_A_B_in_B.pos, zaxis, atol=1e-7)

    t_A_B_in_eps = algebra.transform_move_into_frame(t_A_B, algebra.transform_inv(tA))
    assert tu.tree_close(t_A_B_in_eps.pos, xaxis, atol=1e-7)

    # check that rotation quat is ok
    assert tu.tree_close(maths.rotate(xaxis, t_A_B.rot), yaxis, atol=1e-7)
    assert tu.tree_close(maths.rotate(yaxis, t_A_B.rot), -zaxis, atol=1e-7)

    w, x, y, z = t_A_B.rot
    assert tu.tree_close(t_A_B_in_eps.rot, jnp.array([w, -y, x, z]))

    for seed in range(5):
        tB = Transform.create(rot=maths.quat_random(jax.random.PRNGKey(seed)))
        t_A_B = algebra.transform_mul(tB, algebra.transform_inv(tA))
        t_A_B_in_eps = algebra.transform_move_into_frame(
            t_A_B, algebra.transform_inv(tA)
        )

        w, x, y, z = t_A_B.rot
        assert tu.tree_close(t_A_B_in_eps.rot, jnp.array([w, -y, x, z]))
