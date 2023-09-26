import jax
import jax.numpy as jnp

from ... import base
from .base import GeneratorWithInputExtras
from .base import GeneratorWithInputOutputExtras


def generator_trafo_randomize_positions(
    gen: GeneratorWithInputExtras | GeneratorWithInputOutputExtras,
) -> GeneratorWithInputExtras | GeneratorWithInputOutputExtras:
    def _gen(key, sys):
        key, consume = jax.random.split(key)
        sys = _setup_fn_randomize_positions(consume, sys)
        return gen(key, sys)

    return _gen


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
