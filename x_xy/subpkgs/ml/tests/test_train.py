import jax
import jax.numpy as jnp
import numpy as np
import optax
import tree_utils

import x_xy
from x_xy.subpkgs import ml


def _test_train_rnno_lru(observer_fn):
    example = "test_three_seg_seg2"
    sys = x_xy.io.load_example(example)
    seed = jax.random.PRNGKey(1)
    gen = x_xy.build_generator(
        sys, x_xy.MotionConfig(T=10.0), sizes=1, add_X_imus=True, add_y_relpose=True
    )

    X, y = gen(seed)
    sys_noimu, _ = sys.make_sys_noimu()
    observer = observer_fn(sys_noimu)
    params, state = observer.init(seed, X)

    state = tree_utils.add_batch_dim(state)
    y = observer.apply(params, state, X)[0]

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
        observer,
    )


def test_rnno_gru():
    rnno_fn = lambda sys: ml.make_rnno(sys, 10, 5)
    _test_train_rnno_lru(rnno_fn)


def test_rnno_lru():
    rnno_fn = lambda sys: ml.make_rnno(sys, 10, 5, cell_type="lru")
    _test_train_rnno_lru(rnno_fn)


def test_lru():
    lru_fn = lambda sys: ml.make_lru_observer(sys, 10, 3, 4, 5, 2)
    _test_train_rnno_lru(lru_fn)


def test_train_rnno_lru_nonsocial():
    def gen(_):
        X = dict(a=jnp.ones((1, 1000, 2)), b=dict(c=jnp.zeros((1, 1000, 3))))
        y = jnp.ones((1, 1000, 4))
        return X, y

    for observer in [
        ml.make_rnno(hidden_state_dim=20, message_dim=1),
        ml.make_lru_observer(
            hidden_state_dim_decoder=4,
            hidden_state_dim_encoder=4,
            hidden_state_dim_lru=10,
        ),
    ]:
        X, _ = gen(None)
        params, state = observer.init(jax.random.PRNGKey(1), X)

        state = tree_utils.add_batch_dim(state)
        y = observer.apply(params, state, X)[0]
        assert y.shape == (1, 1000, 4)

        killed = ml.train(gen, 5, observer, callback_kill_if_nan=True)
        assert not killed


def test_checkpointing():
    optimizer = optax.adam(0.1)

    x_xy.setup(unique_id="test_checkpointing_nopause")
    example = "test_three_seg_seg2"
    sys = x_xy.io.load_example(example)
    sys_noimu, _ = sys.make_sys_noimu()
    gen = x_xy.build_generator(
        sys,
        x_xy.MotionConfig(T=10.0),
        sizes=1,
        add_X_imus=True,
        add_y_relpose=True,
        batchsize=1,
        seed=1,
        mode="eager",
    )
    rnno = ml.make_rnno(sys_noimu, hidden_state_dim=20, message_dim=10)

    ml.train(gen, 10, rnno, callback_save_params=True, optimizer=optimizer)
    trained_params_nopause_flat = tree_utils.batch_concat_acme(
        ml.load("~/params/test_checkpointing_nopause.pickle"), 0
    )

    x_xy.setup(unique_id="test_checkpointing_pause")
    ml.train(gen, 10, rnno, callback_kill_after_episode=5, optimizer=optimizer)
    ml.train(
        gen,
        4,
        rnno,
        callback_save_params=True,
        checkpoint="~/.xxy_checkpoints/test_checkpointing_pause",
        optimizer=optimizer,
    )
    trained_params_pause_flat = tree_utils.batch_concat_acme(
        ml.load("~/params/test_checkpointing_pause.pickle"), 0
    )

    np.testing.assert_allclose(trained_params_nopause_flat, trained_params_pause_flat)
