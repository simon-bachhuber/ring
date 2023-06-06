import jax
import jax.numpy as jnp
import numpy as np

import x_xy
from x_xy.algorithms import pd_control


def test_pd_control():
    def evaluate(controller, example, nograv: bool = False):
        sys = x_xy.io.load_example(example)

        if nograv:
            sys = sys.replace(gravity=sys.gravity * 0.0)

        q, xs = x_xy.algorithms.build_generator(
            sys, x_xy.algorithms.RCMG_Config(T=10.0, dang_max=3.0, t_max=0.5)
        )(jax.random.PRNGKey(1))

        jit_step_fn = jax.jit(
            lambda sys, state, tau: x_xy.algorithms.step(sys, state, tau, 1)
        )

        q_reconst = []
        N = len(q)
        state = x_xy.base.State.create(sys)
        controller_state = controller.init(sys, q)

        for _ in range(N):
            controller_state, tau = jax.jit(controller.apply)(
                controller_state, sys, state
            )
            state = jit_step_fn(sys, state, tau)
            q_reconst.append(state.q)

        q_reconst = np.vstack(q_reconst)
        return q, q_reconst

    gains = jnp.array(3 * [10_000] + 5 * [250])
    controller = pd_control(gains, gains * 0.1)
    q, q_reconst = evaluate(controller, "test_control")
    error = jnp.mean(
        x_xy.maths.angle_error(q[:, :4], q_reconst[:, :4]) ** 2
    ) + jnp.mean((q[:, 4:] - q_reconst[:, 4:]) ** 2)
    assert error <= 0.42

    gains = jnp.array(3 * [17] + 3 * [300])
    controller = pd_control(gains, gains)
    q, q_reconst = evaluate(controller, "free")
    error = jnp.sqrt(jnp.mean((q - q_reconst) ** 2))
    assert error <= 0.5

    gains = jnp.array([300.0, 300])
    controller = pd_control(gains, gains)
    q, q_reconst = evaluate(controller, "test_double_pendulum")
    error = jnp.sqrt(jnp.mean((q - q_reconst) ** 2))
    assert error < 0.15
    q, q_reconst = evaluate(controller, "test_double_pendulum", True)
    error = jnp.sqrt(jnp.mean((q - q_reconst) ** 2))
    assert error < 0.1
