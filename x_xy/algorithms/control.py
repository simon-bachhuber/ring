from types import SimpleNamespace

import jax
import jax.numpy as jnp

from x_xy import base, maths, scan

qrel = lambda q1, q2: maths.quat_mul(q1, maths.quat_inv(q2))


def _derivative_quaternion(q, dt):
    axis, angle = maths.quat_to_rot_axis(qrel(q[2:], q[:-2]))
    # axis.shape = (n_timesteps, 3); angle.shape = (n_timesteps,)
    # Thus add singleton dimesions otherwise broadcast error
    dq = axis * angle[:, None] / (2 * dt)
    dq = jnp.vstack((jnp.zeros((3,)), dq, jnp.zeros((3,))))
    return dq


def _derivative(q, dt):
    dq = jnp.vstack(
        (jnp.zeros_like(q[0]), (q[2:] - q[:-2]) / (2 * dt), jnp.zeros_like(q[0]))
    )
    return dq


def _p_control_quaternion(q, q_ref):
    axis, angle = maths.quat_to_rot_axis(qrel(q_ref, q))
    return axis * angle


def pd_control(P: jax.Array = 300.0, D: jax.Array = 300.0):
    """Computes tau using a PD controller. Returns a pair of (init, apply) functions.

    Args:
        P: jax.Array of P gains. Shape: (sys_init.qd_size())
        D: jax.Array of D gains. Shape: (sys_init.qd_size()) where `sys_init` is the
            system that recorded the reference trajectory `q_ref`

    Returns: Pair of (init, apply) functions
        init: (sys, q_ref) -> controller_state
        apply: (controller_state, sys, state) -> controller_state, tau

    Example:
        >>> q_ref = rcmg(sys1)
        >>> cs = pd_control.init(sys1, q_ref)
        >>> for t in range(1000):
        >>>     cs, tau = pd_control.apply(cs, sys2, state)
        >>>     state = dynamics.step(sys2, state, tau)
    """

    def init(sys: base.System, q_ref: jax.Array) -> dict:
        q_qd_ref = {}

        def f(_, __, q_ref_link, name, typ):
            q_ref_link = q_ref_link.T

            if typ == "free":
                dq = _derivative_quaternion(q_ref_link[:, :4], sys.dt)
                qd_ref = jnp.hstack((dq, _derivative(q_ref_link[:, 4:], sys.dt)))
            elif typ == "spherical":
                qd_ref = _derivative_quaternion(q_ref_link, sys.dt)
            else:
                qd_ref = _derivative(q_ref_link, sys.dt)
            q_qd_ref[name] = (q_ref_link, qd_ref)

        scan.tree(sys, f, "qll", q_ref.T, sys.link_names, sys.link_types)
        return {"counter": 0, "q_qd_ref": q_qd_ref}

    def apply(controller_state: dict, sys: base.System, state: base.State) -> jax.Array:
        taus = jnp.zeros((sys.qd_size()))
        q_qd_ref = jax.tree_map(
            lambda arr: jax.lax.dynamic_index_in_dim(
                arr, controller_state["counter"], keepdims=False
            ),
            controller_state["q_qd_ref"],
        )

        def f(_, idx_map, idx, name, typ, q_curr, qd_curr):
            nonlocal taus

            if name not in controller_state["q_qd_ref"]:
                return

            q_ref, qd_ref = q_qd_ref[name]
            if typ == "free":
                P_term = jnp.concatenate(
                    (
                        _p_control_quaternion(q_curr[:4], q_ref[:4]),
                        q_ref[4:] - q_curr[4:],
                    )
                )
            elif typ == "spherical":
                P_term = _p_control_quaternion(q_curr, q_ref)
            else:
                P_term = q_ref - q_curr
            D_term = qd_ref - qd_curr
            tau = P * P_term + D * D_term
            taus = taus.at[idx_map["d"](idx)].set(tau)

        scan.tree(
            sys,
            f,
            "lllqd",
            list(range(sys.num_links())),
            sys.link_names,
            sys.link_types,
            state.q,
            state.qd,
        )

        return {
            "counter": controller_state["counter"] + 1,
            "q_qd_ref": controller_state["q_qd_ref"],
        }, taus

    return SimpleNamespace(init=init, apply=apply)
