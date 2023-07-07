import jax
import jax.numpy as jnp

import x_xy
from x_xy.algorithms.jcalc import _joint_types


def register_rr_joint():
    if "rr" in _joint_types:
        return

    def _rr_transform(q, params):
        def _rxyz_transform(q, _, axis):
            q = jnp.squeeze(q)
            rot = x_xy.maths.quat_rot_axis(axis, q)
            return x_xy.base.Transform.create(rot=rot)

        return _rxyz_transform(q, None, params)

    rr_joint = x_xy.algorithms.JointModel(
        _rr_transform, rcmg_draw_fn=x_xy.algorithms.jcalc._draw_rxyz
    )
    x_xy.algorithms.register_new_joint_type("rr", rr_joint, 1, 0)


def setup_fn_randomize_joint_axes(key, sys: x_xy.base.System) -> x_xy.base.System:
    joint_axes = _draw_random_joint_axis(jax.random.split(key, sys.num_links()))
    return sys.replace(links=sys.links.replace(joint_params=joint_axes))


def setup_fn_randomize_positions(
    key: jax.Array, sys: x_xy.base.System
) -> x_xy.base.System:
    ts = sys.links.transform1

    for i in range(sys.num_links()):
        link = sys.links[i]
        key, new_pos = _draw_pos_uniform(key, link.pos_min, link.pos_max)
        ts = ts.index_set(i, ts[i].replace(pos=new_pos))

    return sys.replace(links=sys.links.replace(transform1=ts))


def _draw_pos_uniform(key, pos_min, pos_max):
    key, c1, c2, c3 = jax.random.split(key, num=4)
    pos = jnp.array(
        [
            jax.random.uniform(c1, minval=pos_min[0], maxval=pos_max[0]),
            jax.random.uniform(c2, minval=pos_min[1], maxval=pos_max[1]),
            jax.random.uniform(c3, minval=pos_min[2], maxval=pos_max[2]),
        ]
    )
    return key, pos


@jax.vmap
def _draw_random_joint_axis(key):
    J = jax.random.uniform(key, (3,), minval=-1.0, maxval=1.0)
    Jnorm = jax.numpy.linalg.norm(J)
    return J / Jnorm
