import numpy as np
import optax
import ring
from ring import ml
from ring import utils
import tree_utils


def test_ring():
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

    ml.train_fn(
        gen,
        5,
        ml.RING(hidden_state_dim=20, message_dim=10, lam=sys_noimu.link_parents),
    )


def test_checkpointing():
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

    ml.train_fn(gen, 10, ringnet, callback_save_params=True, optimizer=optimizer)
    trained_params_nopause_flat = tree_utils.batch_concat_acme(
        utils.pickle_load("~/params/test_checkpointing_nopause.pickle"), 0
    )

    ring.setup(unique_id="test_checkpointing_pause")
    ml.train_fn(gen, 10, ringnet, callback_kill_after_episode=5, optimizer=optimizer)
    ml.train_fn(
        gen,
        4,
        ringnet,
        callback_save_params=True,
        checkpoint="~/.xxy_checkpoints/test_checkpointing_pause",
        optimizer=optimizer,
    )
    trained_params_pause_flat = tree_utils.batch_concat_acme(
        utils.pickle_load("~/params/test_checkpointing_pause.pickle"), 0
    )

    np.testing.assert_allclose(trained_params_nopause_flat, trained_params_pause_flat)
