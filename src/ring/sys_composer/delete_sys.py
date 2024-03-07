from typing import Optional

import jax.numpy as jnp
from ring import base
import tree_utils


def _autodetermine_imu_names(sys: base.System) -> list[str]:
    return sys.findall_imus()


def make_sys_noimu(sys: base.System, imu_link_names: Optional[list[str]] = None):
    "Returns, e.g., imu_attachment = {'imu1': 'seg1', 'imu2': 'seg3'}"
    if imu_link_names is None:
        imu_link_names = _autodetermine_imu_names(sys)
    imu_attachment = {name: sys.parent_name(name) for name in imu_link_names}
    sys_noimu = delete_subsystem(sys, imu_link_names)
    return sys_noimu, imu_attachment


def delete_subsystem(
    sys: base.System, link_name: str | list[str], strict: bool = True
) -> base.System:
    "Cut subsystem starting at `link_name` (inclusive) from tree."
    if isinstance(link_name, list):
        for ln in link_name:
            sys = delete_subsystem(sys, ln, strict)
        return sys

    if not strict:
        try:
            return delete_subsystem(sys, link_name, strict=True)
        except AssertionError:
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

    sys.scan(
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
        link_parents=_reindex_parent_array(sys.link_parents, subsys),
        links=tree_utils.tree_indices(sys.links, jnp.array(keep, dtype=int)),
        link_types=take(sys.link_types),
        link_damping=d,
        link_armature=a,
        link_spring_stiffness=ss,
        link_spring_zeropoint=sz,
        dt=sys.dt,
        geoms=[
            geom.replace(link_idx=idx_map[geom.link_idx])
            for geom in sys.geoms
            if geom.link_idx in keep
        ],
        gravity=sys.gravity,
        integration_method=sys.integration_method,
        mass_mat_iters=sys.mass_mat_iters,
        link_names=take(sys.link_names),
        model_name=sys.model_name,
        omc=take(sys.omc),
    )

    return new_sys.parse()


def _find_subsystem_indices(parents: list[int], k: int) -> list[int]:
    subsys = [k]
    for i, p in enumerate(parents):
        if p in subsys:
            subsys.append(i)
    return subsys


def _idx_map_and_keepers(parents: list[int], subsys: list[int]):
    num_links = len(parents)
    # keep must be in ascending order
    keep = []
    for i in range(num_links):
        if i not in subsys:
            keep.append(i)

    idx_map = dict(zip([-1] + keep, range(-1, len(keep))))
    return idx_map, keep


def _reindex_parent_array(parents: list[int], subsys: list[int]) -> list[int]:
    idx_map, keep = _idx_map_and_keepers(parents, subsys)
    return [idx_map[p] for i, p in enumerate(parents) if i in keep]
