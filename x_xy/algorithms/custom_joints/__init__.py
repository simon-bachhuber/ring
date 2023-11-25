import jax

import x_xy
from x_xy.algorithms.generator.transforms import GeneratorTrafoSetupFn
from x_xy.algorithms.jcalc import _joint_types

from .rr_imp_joint import register_rr_imp_joint
from .rr_imp_joint import setup_fn_randomize_joint_axes_primary_residual
from .rr_joint import register_rr_joint
from .rr_joint import setup_fn_randomize_joint_axes

register_rr_joint()
register_rr_imp_joint()


def _setup_fn(key, sys):
    c1, c2 = jax.random.split(key)
    if "rr" in _joint_types:
        sys = setup_fn_randomize_joint_axes(c1, sys)
    if "rr_imp" in _joint_types:
        sys = setup_fn_randomize_joint_axes_primary_residual(c2, sys)
    return sys


class GeneratorTrafoRandomizeJointAxes(x_xy.GeneratorTrafo):
    def __call__(self, gen):
        return GeneratorTrafoSetupFn(_setup_fn)(gen)
