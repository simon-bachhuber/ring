from . import base
from . import callbacks
from . import ml_utils
from . import optimizer
from . import ringnet
from . import rnno_v1
from . import train
from . import training_loop
from .base import AbstractFilter
from .ml_utils import on_cluster
from .ml_utils import unique_id
from .optimizer import make_optimizer
from .ringnet import RING
from .train import train_fn

_LPF_CUTOFF_FREQ = 10.0


def RNNO(
    output_dim: int,
    return_quats: bool = False,
    params=None,
    eval: bool = True,
    samp_freq: float | None = None,
    v1: bool = False,
    scale_X: bool = True,
    **kwargs,
):
    assert "message_dim" not in kwargs
    assert "link_output_normalize" not in kwargs
    assert "link_output_dim" not in kwargs

    if v1:
        kwargs.update(
            dict(forward_factory=rnno_v1.rnno_v1_forward_factory, output_dim=output_dim)
        )
    else:
        kwargs.update(
            dict(
                message_dim=0,
                link_output_normalize=False,
                link_output_dim=output_dim,
            )
        )

    ringnet = RING(  # noqa: F811
        params=params,
        **kwargs,
    )
    ringnet = base.NoGraph_FilterWrapper(ringnet, quat_normalize=return_quats)
    if scale_X:
        ringnet = base.ScaleX_FilterWrapper(ringnet)
    if eval and return_quats:
        ringnet = base.LPF_FilterWrapper(ringnet, _LPF_CUTOFF_FREQ, samp_freq=samp_freq)
    if return_quats:
        ringnet = base.GroundTruthHeading_FilterWrapper(ringnet)
    return ringnet
