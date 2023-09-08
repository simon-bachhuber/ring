import jax.numpy as jnp
import tree_utils

from x_xy import scan_sys
from x_xy.io import parse_system

from ... import base


def delete_subsystem(sys: base.System, link_name: str | list[str]) -> base.System:
    "Cut subsystem starting at `link_name` (inclusive) from tree."
    if isinstance(link_name, list):
        for ln in link_name:
            sys = delete_subsystem(sys, ln)
        return sys

    assert (
        link_name in sys.link_names
    ), f"link {link_name} not found in {sys.link_names}"

    subsys = _find_subsystem_indices(sys.link_parents, sys.name_to_idx(link_name))
    idx_map, keep = _idx_map_and_keepers(sys.link_parents, subsys)

    def take(list):
        return [ele for i, ele in enumerate(list) if i in keep]

    d, a, ss, sz = [], [], [], []

    def filter_arrays(_, __, damp, arma, stiff, zero, i: int):
        if i in keep:
            d.append(damp)
            a.append(arma)
            ss.append(stiff)
            sz.append(zero)

    scan_sys(
        sys,
        filter_arrays,
        "dddql",
        sys.link_damping,
        sys.link_armature,
        sys.link_spring_stiffness,
        sys.link_spring_zeropoint,
        list(range(sys.num_links())),
    )

    d, a, ss, sz = map(jnp.concatenate, (d, a, ss, sz))

    new_sys = base.System(
        _reindex_parent_array(sys.link_parents, subsys),
        tree_utils.tree_indices(sys.links, jnp.array(keep, dtype=int)),
        take(sys.link_types),
        d,
        a,
        ss,
        sz,
        sys.dt,
        sys.dynamic_geometries,
        [
            geom.replace(link_idx=idx_map[geom.link_idx])
            for geom in sys.geoms
            if geom.link_idx in keep
        ],
        sys.gravity,
        sys.integration_method,
        sys.mass_mat_iters,
        take(sys.link_names),
        sys.model_name,
    )

    return parse_system(new_sys)


def _find_subsystem_indices(parents: list[int], k: int) -> list[int]:
    subsys = [k]
    for i, p in enumerate(parents):
        if p in subsys:
            subsys.append(i)
    return subsys


def _idx_map_and_keepers(parents: list[int], subsys: list[int]):
    num_links = len(parents)
    keep = list(set(range(num_links)) - set(subsys))
    idx_map = dict(zip([-1] + keep, range(-1, len(keep))))
    return idx_map, keep


def _reindex_parent_array(parents: list[int], subsys: list[int]) -> list[int]:
    idx_map, keep = _idx_map_and_keepers(parents, subsys)
    return [idx_map[p] for i, p in enumerate(parents) if i in keep]
