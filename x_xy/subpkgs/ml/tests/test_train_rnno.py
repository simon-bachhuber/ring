import jax
import tree_utils

import x_xy
from x_xy.algorithms.generator import transforms
from x_xy.subpkgs import ml
from x_xy.subpkgs import sys_composer


def pipeline_make_generator(
    config: x_xy.RCMG_Config,
    bs: int,
    sys: x_xy.System,
):
    sys_noimu, _ = sys_composer.make_sys_noimu(sys)

    gen = x_xy.GeneratorPipe(
        transforms.GeneratorTrafoIMU(),
        transforms.GeneratorTrafoRelPose(sys_noimu),
        x_xy.GeneratorTrafoRemoveOutputExtras(),
        x_xy.GeneratorTrafoRemoveInputExtras(sys),
    )(config)

    return x_xy.batch_generator(gen, bs)


def test_rnno():
    example = "test_three_seg_seg2"
    sys = x_xy.io.load_example(example)
    seed = jax.random.PRNGKey(1)
    gen = pipeline_make_generator(x_xy.RCMG_Config(T=10.0), 1, sys)

    X, y = gen(seed)
    sys_noimu, _ = sys_composer.make_sys_noimu(sys)
    rnno = ml.make_rnno(sys_noimu, 10, 2)
    params, state = rnno.init(seed, X)

    state = tree_utils.add_batch_dim(state)
    y = rnno.apply(params, state, X)[0]

    for name in sys_noimu.link_names:
        assert name in X
        for sensor in ["acc", "gyr"]:
            assert sensor in X[name]
            assert X[name][sensor].shape == (1, 1000, 3)

        p = sys_noimu.link_parents[sys_noimu.name_to_idx(name)]
        if p == -1:
            assert name not in y
        else:
            assert name in y
            assert y[name].shape == (1, 1000, 4)

    ml.train(
        gen,
        5,
        rnno,
    )
