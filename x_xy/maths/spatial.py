"""
Implements `Table 1` from
    `A Beginner's Guide to 6-D Vectors (Part 2)`
    by Roy Featherstone.

`A small but sufficient set of spatial arithemtic operations.`
"""


import jax.numpy as jnp


def rx(theta):
    """
    [
        1  0 0
        0  c s
        0 -s c
    ]
    where c = cos(theta)
          s = sin(theta)
    """
    s, c = jnp.sin(theta), jnp.cos(theta)
    return jnp.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def ry(theta):
    """
    [
        c 0 -s
        0 1  0
        s 0  c
    ]
    where c = cos(theta)
          s = sin(theta)
    """
    s, c = jnp.sin(theta), jnp.cos(theta)
    return jnp.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def rz(theta):
    """
    [
         c s 0
        -s c 0
         0 0 1
    ]
    where c = cos(theta)
          s = sin(theta)
    """
    s, c = jnp.sin(theta), jnp.cos(theta)
    return jnp.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def cross(r):
    assert r.shape == (3,)
    return jnp.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])


def quadrants(aa=None, ab=None, ba=None, bb=None, default=jnp.zeros):
    M = default((6, 6))
    if aa is not None:
        M = M.at[:3, :3].set(aa)
    if ab is not None:
        M = M.at[:3, 3:].set(ab)
    if ba is not None:
        M = M.at[3:, :3].set(ba)
    if bb is not None:
        M = M.at[3:, 3:].set(bb)
    return M


def crm(v):
    assert v.shape == (6,)
    return quadrants(cross(v[:3]), ba=cross(v[3:]), bb=cross(v[:3]))


def crf(v):
    return -crm(v).T


def _rotxyz(E):
    return quadrants(E, bb=E)


def rotx(theta):
    return _rotxyz(rx(theta))


def roty(theta):
    return _rotxyz(ry(theta))


def rotz(theta):
    return _rotxyz(rz(theta))


def xlt(r):
    assert r.shape == (3,)
    return quadrants(jnp.eye(3), ba=-cross(r), bb=jnp.eye(3))


def X_transform(E, r):
    return _rotxyz(E) @ xlt(r)


def mcI(m, c, Ic):
    assert c.shape == (3,)
    assert Ic.shape == (3, 3)
    return quadrants(
        Ic - m * cross(c) @ cross(c), m * cross(c), -m * cross(c), m * jnp.eye(3)
    )


def XtoV(X):
    assert X.shape == (6, 6)
    return 0.5 * jnp.array(
        [
            [X[1, 2] - X[2, 1]],
            [X[2, 0] - X[0, 2]],
            [X[0, 1] - X[1, 0]],
            [X[4, 2] - X[5, 1]],
            [X[5, 0] - X[3, 2]],
            [X[3, 1] - X[4, 0]],
        ]
    )
