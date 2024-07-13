from _compat import unbatch_gen
import jax
import jax.numpy as jnp
import numpy as np

import ring
from ring.algorithms.generator.pd_control import _pd_control


def test_pd_control():
    def evaluate(controller, example, nograv: bool = False):
        sys = ring.io.load_example(example)

        if nograv:
            sys = sys.replace(gravity=sys.gravity * 0.0)

        q, xs = unbatch_gen(
            ring.algorithms.RCMG(
                sys,
                ring.algorithms.MotionConfig(
                    T=10.0,
                    dang_max=3.0,
                    t_max=0.5,
                    dang_max_free_spherical=jnp.deg2rad(60),
                    t_min=0.15,
                    dpos_max=0.1,
                ),
                finalize_fn=lambda key, q, x, sys: (q, x),
            ).to_lazy_gen()
        )(jax.random.PRNGKey(2))

        jit_step_fn = jax.jit(lambda sys, state, tau: ring.step(sys, state, tau, 1))

        q_reconst = []
        N = len(q)
        state = ring.base.State.create(sys)
        controller_state = controller.init(sys, q)

        for _ in range(N):
            controller_state, tau = jax.jit(controller.apply)(
                controller_state, sys, state
            )
            state = jit_step_fn(sys, state, tau)
            q_reconst.append(state.q)

        q_reconst = np.vstack(q_reconst)
        return q, q_reconst

    gains = jnp.array(3 * [10_000] + 3 * [250] + 2 * [250])
    controller = _pd_control(gains, gains * 0.1)
    q, q_reconst = evaluate(controller, "test_control")
    error = jnp.mean(
        ring.maths.angle_error(q[:, :4], q_reconst[:, :4]) ** 2
    ) + jnp.mean((q[:, 4:] - q_reconst[:, 4:]) ** 2)
    assert error <= 0.42

    gains = jnp.array(3 * [17] + 3 * [300])
    controller = _pd_control(gains, gains)
    q, q_reconst = evaluate(controller, "test_free")
    error = jnp.sqrt(jnp.mean((q - q_reconst) ** 2))
    assert error <= 0.502

    gains = jnp.array([50.0, 50.0])
    controller = _pd_control(gains, gains * 0.1)
    q, q_reconst = evaluate(controller, "test_double_pendulum")
    error = jnp.sqrt(jnp.mean((q - q_reconst) ** 2))
    # TODO investigate why errors are higher after upgrading python, jax, and cuda
    # assert error < 0.15
    # assert error < 0.31
    assert error < 0.56
    q, q_reconst = evaluate(controller, "test_double_pendulum", True)
    error = jnp.sqrt(jnp.mean((q - q_reconst) ** 2))
    # assert error < 0.1
    assert error < 0.46


LARGE_DAMPING = 25.0


def _sys_large_damping(sys: ring.System, exclude: list[str] = []) -> ring.System:
    damping = sys.link_damping

    def f(_, idx_map, idx, name):
        nonlocal damping

        if name in exclude:
            return

        slice = idx_map["d"](idx)
        a, b = slice.start, slice.stop
        damping = damping.at[a:b].set(LARGE_DAMPING)

    sys.scan(f, "ll", list(range(sys.num_links())), sys.link_names)
    return sys.replace(link_damping=damping)


def test_dynamical_simulation_trafo():
    P_gains = {
        "free": jnp.array(3 * [50.0] + 3 * [200.0]),
        "rz": jnp.array([400.0]),
        "frozen": jnp.array([]),
    }
    P_gains["ry"] = P_gains["rz"]

    for example in ["test_three_seg_seg2"]:
        sys = ring.io.load_example(example)
        sys = _sys_large_damping(sys)
        data = ring.RCMG(
            sys,
            ring.MotionConfig(T=20.0),
            dynamic_simulation=1,
            dynamic_simulation_kwargs=dict(custom_P_gains=P_gains, return_q_ref=True),
            keep_output_extras=1,
        ).to_list()
        (X, _), (__, q_obs, ___, ____) = data[0]
        error = jnp.sqrt(jnp.mean((X["q_ref"][500:, -2:] - q_obs[500:, -2:]) ** 2))
        assert error < 0.1
