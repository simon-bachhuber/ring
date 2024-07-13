import jax
import jax.numpy as jnp

from ring import base
from ring import maths


def _setup_fn_randomize_positions(key: jax.Array, sys: base.System) -> base.System:
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


def _setup_fn_randomize_transform1_rot(
    key, sys, maxval: float, not_imus: bool = True
) -> base.System:
    new_transform1 = sys.links.transform1.replace(
        rot=maths.quat_random(key, (sys.num_links(),), maxval=maxval)
    )
    if not_imus:
        imus = [name for name in sys.link_names if name[:3] == "imu"]
        new_rot = new_transform1.rot
        for imu in imus:
            new_rot = new_rot.at[sys.name_to_idx(imu)].set(jnp.array([1.0, 0, 0, 0]))
        new_transform1 = new_transform1.replace(rot=new_rot)
    return sys.replace(links=sys.links.replace(transform1=new_transform1))
