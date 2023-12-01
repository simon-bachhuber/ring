import jax

import x_xy
from x_xy.algorithms.generator.transforms import GeneratorTrafoSetupFn
from x_xy.algorithms.jcalc import _joint_types

from .rr_imp_joint import register_rr_imp_joint
from .rr_joint import register_rr_joint

register_rr_joint()
register_rr_imp_joint()
