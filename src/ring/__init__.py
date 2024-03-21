from . import algebra
from . import algorithms
from . import base
from . import io
from . import maths
from . import ml
from . import rendering
from . import sim2real
from . import spatial
from . import sys_composer
from . import utils
from .algorithms import join_motionconfigs
from .algorithms import JointModel
from .algorithms import MotionConfig
from .algorithms import RCMG
from .algorithms import register_new_joint_type
from .algorithms import step
from .base import State
from .base import System
from .base import Transform
from .ml import RING

_TRAIN_TIMING_START = None
_UNIQUE_ID = None


def setup(
    rr_joint_kwargs: None | dict = dict(),
    rr_imp_joint_kwargs: None | dict = dict(),
    suntay_joint_kwargs: None | dict = None,
    train_timing_start: None | float = None,
    unique_id: None | str = None,
):
    import time

    from ring.algorithms import custom_joints

    global _TRAIN_TIMING_START
    global _UNIQUE_ID

    if rr_joint_kwargs is not None:
        custom_joints.register_rr_joint(**rr_joint_kwargs)

    if rr_imp_joint_kwargs is not None:
        custom_joints.register_rr_imp_joint(**rr_imp_joint_kwargs)

    if suntay_joint_kwargs is not None:
        custom_joints.register_suntay(**suntay_joint_kwargs)

    if _TRAIN_TIMING_START is None:
        _TRAIN_TIMING_START = time.time()

    if train_timing_start is not None:
        _TRAIN_TIMING_START = train_timing_start

    if _UNIQUE_ID is None:
        _UNIQUE_ID = hex(hash(time.time()))

    if unique_id is not None:
        _UNIQUE_ID = unique_id


setup()
