import x_xy
from x_xy.algorithms.generator.transforms import GeneratorTrafoSetupFn

from .rr_imp_joint import register_rr_imp_joint
from .rr_imp_joint import setup_fn_randomize_joint_axes_primary_residual
from .rr_joint import register_rr_joint
from .rr_joint import setup_fn_randomize_joint_axes


def _setup_fn(key, sys):
    n_joint_params = sys.links.joint_params.shape[-1]
    if n_joint_params == 3:
        sys = setup_fn_randomize_joint_axes(key, sys)
    elif n_joint_params == 6:
        sys = setup_fn_randomize_joint_axes_primary_residual(key, sys)
    else:
        raise Exception("Panic :)")
    return sys


class GeneratorTrafoRandomizeJointAxes(x_xy.GeneratorTrafo):
    def __call__(self, gen):
        return GeneratorTrafoSetupFn(_setup_fn)(gen)
