from pathlib import Path

import numpy as np
import optax
import tree_utils

import ring
from ring import ml
from ring import utils


def _load_gen_lam():
    example = "test_three_seg_seg2"
    sys = ring.io.load_example(example)
    sys_noimu = sys.make_sys_noimu()[0]
    gen = ring.RCMG(
        sys,
        ring.MotionConfig(T=10.0),
        add_X_imus=1,
        add_y_relpose=1,
        add_y_rootincl=1,
        use_link_number_in_Xy=1,
    ).to_lazy_gen()
    gen = ring.algorithms.GeneratorTrafoExpandFlatten(gen)
    lam = sys_noimu.link_parents
    return gen, lam


def test_rnno():
    gen, lam = _load_gen_lam()

    N = len(lam)
    ml.train_fn(
        gen,
        5,
        # .unwrapped to get ride of the `GroundtruthHeadingWrapper`
        ml.RNNO(N * 4, return_quats=True, eval=False, hidden_state_dim=20).unwrapped,
    )


def test_ring():
    gen, lam = _load_gen_lam()

    ml.train_fn(
        gen,
        5,
        ml.RING(hidden_state_dim=20, message_dim=10, lam=lam),
    )


def _remove_file_if_exists(path: str) -> None:
    Path(path).expanduser().unlink(missing_ok=True)


def test_checkpointing():
    _remove_file_if_exists("~/params/test_checkpointing_nopause.pickle")
    _remove_file_if_exists("~/params/test_checkpointing_pause.pickle")
    _remove_file_if_exists("~/.ring_checkpoints/test_checkpointing_pause.pickle")

    optimizer = optax.adam(0.1)

    ring.setup(unique_id="test_checkpointing_nopause")
    example = "test_three_seg_seg2"
    sys = ring.io.load_example(example)
    sys_noimu, _ = sys.make_sys_noimu()
    gen = ring.RCMG(
        sys,
        ring.MotionConfig(T=10.0),
        add_X_imus=True,
        add_y_relpose=True,
        add_y_rootincl=True,
        use_link_number_in_Xy=1,
    ).to_eager_gen()
    gen = ring.algorithms.GeneratorTrafoExpandFlatten(gen)

    ringnet = ml.RING(hidden_state_dim=20, message_dim=10, lam=sys_noimu.link_parents)

    ml.train_fn(
        gen,
        10,
        ringnet,
        callback_save_params=True,
        optimizer=optimizer,
        callback_create_checkpoint=False,
    )
    trained_params_nopause_flat = tree_utils.batch_concat_acme(
        utils.pickle_load("~/params/test_checkpointing_nopause.pickle"), 0
    )

    ring.setup(unique_id="test_checkpointing_pause")
    ml.train_fn(
        gen,
        10,
        ringnet,
        optimizer=optimizer,
        callback_create_checkpoint=True,
        callback_kill_after_episode=5,
    )
    ml.train_fn(
        gen,
        4,
        ringnet,
        callback_save_params=True,
        checkpoint="~/.ring_checkpoints/test_checkpointing_pause",
        optimizer=optimizer,
    )
    trained_params_pause_flat = tree_utils.batch_concat_acme(
        utils.pickle_load("~/params/test_checkpointing_pause.pickle"), 0
    )

    np.testing.assert_allclose(trained_params_nopause_flat, trained_params_pause_flat)
