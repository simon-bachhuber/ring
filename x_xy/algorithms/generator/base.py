from typing import Sequence

import jax
import jax.numpy as jnp

from ... import base
from ...scan import scan_sys
from ..jcalc import _joint_types
from ..jcalc import RCMG_Config
from ..kinematics import forward_kinematics_transforms
from .transforms import GeneratorTrafoFinalizeFn
from .transforms import GeneratorTrafoRandomizePositions
from .transforms import GeneratorTrafoSetupFn
from .types import FINALIZE_FN
from .types import Generator
from .types import GeneratorTrafo
from .types import GeneratorWithInputExtras
from .types import GeneratorWithInputOutputExtras
from .types import GeneratorWithOutputExtras
from .types import OutputExtras
from .types import PRNGKey
from .types import SETUP_FN


def build_generator(
    sys: base.System,
    config: RCMG_Config = RCMG_Config(),
    setup_fn: SETUP_FN = lambda key, sys: sys,
    finalize_fn: FINALIZE_FN = lambda key, q, x, sys: (q, x),
    randomize_positions: bool = False,
) -> Generator:
    return GeneratorPipe(
        GeneratorTrafoSetupFn(setup_fn),
        GeneratorTrafoRandomizePositions()
        if randomize_positions
        else (lambda gen: gen),
        GeneratorTrafoFinalizeFn(finalize_fn),
        GeneratorTrafoRemoveInputExtras(sys),
        GeneratorTrafoRemoveOutputExtras(),
    )(config)


def _generator_with_extras(
    config: RCMG_Config,
) -> GeneratorWithInputOutputExtras:
    def generator(key: PRNGKey, sys: base.System) -> OutputExtras:
        if config.cor:
            sys = _replace_free_with_cor(sys)

        key_start = key
        # build generalized coordintes vector `q`
        q_list = []

        def draw_q(key, __, link_type, joint_params):
            if key is None:
                key = key_start
            key, key_t, key_value = jax.random.split(key, 3)
            draw_fn = _joint_types[link_type].rcmg_draw_fn
            if draw_fn is None:
                raise Exception(f"The joint type {link_type} has no draw fn specified.")
            q_link = draw_fn(config, key_t, key_value, joint_params)
            # even revolute and prismatic joints must be 2d arrays
            q_link = q_link if q_link.ndim == 2 else q_link[:, None]
            q_list.append(q_link)
            return key

        keys = scan_sys(sys, draw_q, "ll", sys.link_types, sys.links.joint_params)
        # stack of keys; only the last key is unused
        key = keys[-1]

        q = jnp.concatenate(q_list, axis=1)

        # do forward kinematics
        x, _ = jax.vmap(forward_kinematics_transforms, (None, 0))(sys, q)

        Xy = ({}, {})
        return Xy, (key, q, x, sys)

    return generator


class GeneratorPipe:
    def __init__(self, *gen_trafos: Sequence[GeneratorTrafo]):
        self._gen_trafos = gen_trafos

    def __call__(
        self, config: RCMG_Config
    ) -> (
        GeneratorWithInputOutputExtras
        | GeneratorWithOutputExtras
        | GeneratorWithInputExtras
        | Generator
    ):
        gen = _generator_with_extras(config)
        for trafo in self._gen_trafos:
            gen = trafo(gen)
        return gen


class GeneratorTrafoRemoveInputExtras(GeneratorTrafo):
    def __init__(self, sys: base.System):
        self.sys = sys

    def __call__(
        self,
        gen: GeneratorWithInputExtras | GeneratorWithInputOutputExtras,
    ) -> Generator | GeneratorWithOutputExtras:
        def _gen(key):
            return gen(key, self.sys)

        return _gen


class GeneratorTrafoRemoveOutputExtras(GeneratorTrafo):
    def __call__(
        self,
        gen: GeneratorWithOutputExtras | GeneratorWithInputOutputExtras,
    ) -> Generator | GeneratorWithInputExtras:
        def _gen(*args):
            return gen(*args)[0]

        return _gen


def _replace_free_with_cor(sys: base.System) -> base.System:
    # checks
    for i, p in enumerate(sys.link_parents):
        link_type = sys.link_types[i]
        if p == -1:
            assert link_type == "free"
        if link_type == "free":
            assert p == -1

    return sys.replace(
        link_types=["cor" if typ == "free" else typ for typ in sys.link_types]
    )
