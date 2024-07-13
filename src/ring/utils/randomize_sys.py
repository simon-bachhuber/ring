"""Randomization by modifying System and MotionConfig objects before building
generator."""

from dataclasses import replace
import itertools
from typing import Optional
import warnings

import jax.numpy as jnp
from ring import base
from ring.algorithms import jcalc
from ring.algorithms.generator import types


def _find_children(lam: list[int], body: int) -> list[int]:

    children = []

    def _children(body: int) -> None:
        for i in range(len(lam)):
            if lam[i] == body:
                children.append(i)
                _children(i)

    _children(body)
    return children


def _find_root_of_subsys_that_contains_body(sys: base.System, body: str) -> str:
    body_i = sys.name_to_idx(body)
    for i, p in enumerate(sys.link_parents):
        if p == -1:
            if body_i == i or body_i in _find_children(sys.link_parents, i):
                return sys.idx_to_name(i)


def _assign_anchors_to_subsys(sys: base.System, anchors: list[str]) -> list[list[str]]:
    anchors_per_subsys = []
    for i, p in enumerate(sys.link_parents):
        if p == -1:
            link_idxs_subsys = [i] + _find_children(sys.link_parents, i)
            link_names_subsys = [sys.idx_to_name(i) for i in link_idxs_subsys]
            anchors_this_subsys = [
                name for name in anchors if name in link_names_subsys
            ]
            if len(anchors_this_subsys) == 0:
                anchors_this_subsys = [sys.idx_to_name(i)]
            anchors_per_subsys.append(anchors_this_subsys)
    return anchors_per_subsys


def _morph_extract_subsys(sys: base.System, anchor: str):
    root = _find_root_of_subsys_that_contains_body(sys, anchor)
    roots = sys.findall_bodies_to_world(names=True)
    subsys = sys.delete_system(list(set(roots) - set([root])))
    return subsys.morph_system(new_anchor=anchor)


def randomize_anchors(
    sys: base.System, anchors: Optional[list[str]] = None
) -> list[base.System]:

    if anchors is None:
        anchors = sys.findall_segments()

    anchors = _assign_anchors_to_subsys(sys, anchors)
    syss = []
    for anchors_subsys in itertools.product(*anchors):
        sys_mod = _morph_extract_subsys(sys, anchors_subsys[0])
        for anchor_subsys in anchors_subsys[1:]:
            sys_mod = sys_mod.inject_system(_morph_extract_subsys(sys, anchor_subsys))
        syss.append(sys_mod)

    return syss


_WARN_HZ_Threshold: float = 40.0


def randomize_hz(
    sys: list[base.System],
    configs: list[jcalc.MotionConfig],
    sampling_rates: list[float],
) -> tuple[list[base.System], list[jcalc.MotionConfig]]:
    Ts = [c.T for c in configs]
    assert len(set(Ts)), f"Time length between configs does not agree {Ts}"
    T_global = Ts[0]

    for hz in sampling_rates:
        if hz < _WARN_HZ_Threshold:
            warnings.warn(
                "The sampling rate {hz} is below the warning threshold of "
                f"{_WARN_HZ_Threshold}. This might lead to NaNs."
            )

    sys_out, configs_out = [], []
    for _sys in sys:
        for _config in configs:
            for hz in sampling_rates:
                dt = 1 / hz
                T = (T_global / _sys.dt) * dt

                sys_out.append(_sys.replace(dt=dt))
                configs_out.append(replace(_config, T=T))
    return sys_out, configs_out


def randomize_hz_finalize_fn_factory(finalize_fn_user: types.FINALIZE_FN | None):
    def finalize_fn(key, q, x, sys: base.System):
        X, y = {}, {}
        if finalize_fn_user is not None:
            X, y = finalize_fn_user(key, q, x, sys)

        assert "dt" not in X
        X["dt"] = jnp.array([sys.dt], dtype=jnp.float32)

        return X, y

    return finalize_fn
