from . import base
from . import callbacks
from . import ml_utils
from . import optimizer
from . import ringnet
from . import train
from . import training_loop
from .base import AbstractFilter
from .ml_utils import on_cluster
from .ml_utils import unique_id
from .optimizer import make_optimizer
from .ringnet import RING
from .train import train_fn


def RING_ICML24(**kwargs):
    from pathlib import Path

    params = Path(__file__).parent.joinpath("params/0x13e3518065c21cd8.pickle")
    ringnet = RING(params=params, **kwargs)  # noqa: F811
    ringnet = base.ScaleX_FilterWrapper(ringnet)
    ringnet = base.LPF_FilterWrapper(ringnet, 10.0, samp_freq=None)
    ringnet = base.GroundTruthHeading_FilterWrapper(ringnet)
    return ringnet
