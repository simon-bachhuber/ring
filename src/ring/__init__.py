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


def RING(lam: list[int] | None, Ts: float | None, **kwargs) -> ml.AbstractFilter:
    """Creates the RING network.

    Params:
        lam: parent array, if `None` must be given via `ringnet.apply(..., lam=lam)`
        Ts : sampling interval of IMU data; time delta in seconds

    Returns:
        ring.ml.AbstractFilter: An instantiation of `ring.ml.ringnet.RING` with trained
            parameters.

    Examples:
        >>> import ring
        >>> import numpy as np
        >>>
        >>> T  : int       = 30        # sequence length     [s]
        >>> Ts : float     = 0.01      # sampling interval   [s]
        >>> B  : int       = 1         # batch size
        >>> lam: list[int] = [0, 1, 2] # parent array
        >>> N  : int       = len(lam)  # number of bodies
        >>> T_i: int       = int(T/Ts) # number of timesteps
        >>>
        >>> X = np.zeros((B, T_i, N, 9))
        >>> # where X is structured as follows:
        >>> # X[..., :3]  = acc
        >>> # X[..., 3:6] = gyr
        >>> # X[..., 6:9] = jointaxis
        >>>
        >>> # let's assume we have an IMU on each outer segment of the
        >>> # three-segment kinematic chain
        >>> X[:, :, 0, :3]  = acc_segment1
        >>> X[:, :, 2, :3]  = acc_segment3
        >>> X[:, :, 0, 3:6] = gyr_segment1
        >>> X[:, :, 2, 3:6] = gyr_segment3
        >>>
        >>> ringnet = ring.RING(lam, Ts)
        >>>
        >>> yhat, _ = ringnet.apply(X)
        >>> # yhat : unit quaternions, shape = (B, T_i, N, 4)
        >>> # yhat[b, :, i] is the orientation from body `i` to parent body `lam[i]`
        >>>
        >>> # use `jax.jit` to compile the forward pass
        >>> jit_apply = jax.jit(ringnet.apply)
        >>> yhat, _ = jit_apply(X)
        >>>
        >>> # manually pass in and out the hidden state like so
        >>> initial_state = None
        >>> yhat, state = ringnet.apply(X, state=initial_state)
        >>> # state: final hidden state, shape = (B, N, 2*H)
    """
    from pathlib import Path
    import warnings

    config = dict(
        use_100Hz_RING=True,
        use_lpf=True,
        lpf_cutoff_freq=ml._LPF_CUTOFF_FREQ,
    )
    config.update(kwargs)

    if Ts is not None and (Ts > (1 / 40) or Ts < (1 / 200)):
        warnings.warn(
            "RING was only trained on sampling rates between 40 to 200 Hz "
            f"but found {1 / Ts}Hz"
        )

    if Ts is not None and Ts == 0.01 and config["use_100Hz_RING"]:
        # this set of parameters was trained exclusively on 100Hz data; it also
        # expects F=9 features per node and not F=10 where the last features is
        # the sampling interval Ts
        params = Path(__file__).parent.joinpath("ml/params/0x1d76628065a71e0f.pickle")
        add_Ts = False
    else:
        # this set of parameters was trained on sampling rates from 40 to 200 Hz
        params = Path(__file__).parent.joinpath("ml/params/0x13e3518065c21cd8.pickle")
        add_Ts = True

    ringnet = ml.RING(
        params=params,
        lam=None if lam is None else tuple(lam),
        jit=config.pop("jit", False),
        name="RING",
    )
    ringnet = ml.base.ScaleX_FilterWrapper(ringnet)
    if config["use_lpf"]:
        ringnet = ml.base.LPF_FilterWrapper(
            ringnet,
            config["lpf_cutoff_freq"],
            samp_freq=None if Ts is None else 1 / Ts,
            quiet=True,
        )
    ringnet = ml.base.GroundTruthHeading_FilterWrapper(ringnet)
    if add_Ts:
        ringnet = ml.base.AddTs_FilterWrapper(ringnet, Ts)
    return ringnet


_TRAIN_TIMING_START = None
_UNIQUE_ID = None


def setup(
    rr_joint_kwargs: None | dict = dict(),
    rr_imp_joint_kwargs: None | dict = dict(),
    rsaddle_joint_kwargs: None | dict = dict(),
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

    if rsaddle_joint_kwargs is not None:
        custom_joints.register_rsaddle_joint(**rsaddle_joint_kwargs)

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
