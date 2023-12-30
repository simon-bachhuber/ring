from . import algebra
from . import algorithms
from . import base
from . import io
from . import maths
from . import rendering
from . import utils
from .algebra import transform_inv
from .algebra import transform_mul
from .algorithms import batch_generators_eager
from .algorithms import batch_generators_eager_to_list
from .algorithms import batch_generators_lazy
from .algorithms import batched_generator_from_list
from .algorithms import batched_generator_from_paths
from .algorithms import build_generator
from .algorithms import concat_configs
from .algorithms import forward_kinematics
from .algorithms import GeneratorPipe
from .algorithms import GeneratorTrafo
from .algorithms import GeneratorTrafoRandomizePositions
from .algorithms import GeneratorTrafoRemoveInputExtras
from .algorithms import GeneratorTrafoRemoveOutputExtras
from .algorithms import get_joint_model
from .algorithms import imu
from .algorithms import inverse_kinematics_endeffector
from .algorithms import joint_axes
from .algorithms import JointModel
from .algorithms import make_normalizer_from_generator
from .algorithms import RCMG_Config
from .algorithms import register_new_joint_type
from .algorithms import rel_pose
from .algorithms import step
from .base import State
from .base import System
from .base import Transform
from .io import load_example
from .io import load_sys_from_str
from .io import load_sys_from_xml
from .io import save_sys_to_str
from .io import save_sys_to_xml
from .rendering import render
from .rendering import render_prediction
from .scan import scan_sys


def setup(
    rr_joint_kwargs: None | dict = dict(),
    rr_imp_joint_kwargs: None | dict = dict(),
    suntay_joint_kwargs: None | dict = None,
):
    from x_xy.algorithms import custom_joints

    if rr_joint_kwargs is not None:
        custom_joints.register_rr_joint(**rr_joint_kwargs)

    if rr_imp_joint_kwargs is not None:
        custom_joints.register_rr_imp_joint(**rr_imp_joint_kwargs)

    if suntay_joint_kwargs is not None:
        custom_joints.register_suntay(**suntay_joint_kwargs)


setup()
