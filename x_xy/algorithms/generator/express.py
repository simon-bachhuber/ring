from ...base import System
from ..jcalc import RCMG_Config
from .base import GeneratorPipe
from .base import GeneratorTrafoRemoveInputExtras
from .base import GeneratorTrafoRemoveOutputExtras
from .batch import batch_generators_eager_to_list
from .transforms import GeneratorTrafoIMU
from .transforms import GeneratorTrafoRandomizePositions
from .transforms import GeneratorTrafoRelPose


def express_Ximu_Yrelpose_data(
    sys: System,
    config: RCMG_Config = RCMG_Config(),
    n_sequences: int = 1,
    seed: int = 1,
) -> list:
    from x_xy.subpkgs import sys_composer

    sys_noimu, _ = sys_composer.make_sys_noimu(sys)

    gen = GeneratorPipe(
        GeneratorTrafoRandomizePositions(),
        GeneratorTrafoIMU(),
        GeneratorTrafoRelPose(sys_noimu),
        GeneratorTrafoRemoveInputExtras(sys),
        GeneratorTrafoRemoveOutputExtras(),
    )(config)
    return batch_generators_eager_to_list(gen, n_sequences, seed)
