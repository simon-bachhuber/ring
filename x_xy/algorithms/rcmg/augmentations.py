from xml.etree import ElementTree

import jax
import jax.numpy as jnp

import x_xy
from x_xy.io.xml.from_xml import _initial_setup, _load_xml


def register_rr_joint():
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


def setup_fn_randomize_positions(xml_path: str, prefix: str = ""):
    xml_tree = ElementTree.fromstring(_load_xml(xml_path))
    worldbody = _initial_setup(xml_tree)

    pos_min_max = {}

    def process_body(body: ElementTree):
        name = body.attrib["name"]
        pos_min = body.attrib.get("pos_min", None)
        pos_max = body.attrib.get("pos_max", None)

        assert (pos_min is None and pos_max is None) or (
            pos_min is not None and pos_max is not None
        ), (
            f"In link {name} found only one of `pos_min` and `pos_max`, but"
            " requires either both or none"
        )

        if pos_min is not None:
            pos_min_max[prefix + name] = (pos_min, pos_max)

        for subbodies in body.findall("body"):
            process_body(subbodies)

    for body in worldbody.findall("body"):
        process_body(body)

    def _randomize_positions(key: jax.Array, sys: x_xy.base.System) -> x_xy.base.System:
        ts = sys.links.transform1

        for name, (pos_min, pos_max) in pos_min_max.items():
            i = sys.name_to_idx(name)
            key, new_pos = _draw_pos_uniform(key, pos_min, pos_max)
            ts = ts.index_set(i, ts[i].replace(pos=new_pos))

        return sys.replace(links=sys.links.replace(transform1=ts))

    return _randomize_positions


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
