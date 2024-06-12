from types import SimpleNamespace
from typing import Optional

from flax import struct
import jax
import jax.numpy as jnp

from ring import base
from ring.algorithms import dynamics
from ring.algorithms import jcalc


@struct.dataclass
class PDControllerState:
    i: int
    q_ref_as_dict: dict
    qd_ref_as_dict: dict
    P_gains: dict
    D_gains: dict


def _pd_control(P: jax.Array, D: Optional[jax.Array] = None):
    """Computes tau using a PD controller. Returns a pair of (init, apply) functions.

    NOTE: Gains around ~10_000 are good for spherical joints, everything else ~250-300
    works just fine. Damping should be about 2500 for spherical joints, and
    about 25 for everything else.

    Args:
        P: jax.Array of P gains. Shape: (sys_init.qd_size())
        D: jax.Array of D gains. Shape: (sys_init.qd_size()) where `sys_init` is the
            system that recorded the reference trajectory `q_ref`
            If not given, then no D control is applied.

    Returns: Pair of (init, apply) functions
        init: (sys, q_ref) -> controller_state
        apply: (controller_state, sys, state) -> controller_state, tau

    Example:
        >>> gains = jnp.array([250.0] * sys1.qd_size())
        >>> controller = pd_control(gains, gains)
        >>> q_ref = rcmg(sys1)
        >>> cs = controller.init(sys1, q_ref)
        >>> for t in range(1000):
        >>>     cs, tau = controller.apply(cs, sys2, state)
        >>>     state = dynamics.step(sys2, state, tau)
    """

    def init(sys: base.System, q_ref: jax.Array) -> dict:
        assert sys.q_size() == q_ref.shape[1], f"q_ref.shape = {q_ref.shape}"
        assert sys.qd_size() == P.size
        if D is not None:
            assert sys.qd_size() == D.size

        q_ref_as_dict = {}
        qd_ref_as_dict = {}
        P_as_dict = {}
        D_as_dict = {}

        def f(_, __, q_ref_link, name, typ, P_link, D_link):
            P_as_dict[name] = P_link
            q_ref_link = q_ref_link.T
            q_ref_as_dict[name] = q_ref_link

            if D is not None:
                qd_from_q = jcalc.get_joint_model(typ).qd_from_q
                if qd_from_q is None:
                    raise NotImplementedError(
                        f"Please specify `JointModel.qd_from_q` for joint type `{typ}`"
                    )
                qd_ref_as_dict[name] = qd_from_q(q_ref_link, sys.dt)
                D_as_dict[name] = D_link

        sys.scan(
            f,
            "qlldd",
            q_ref.T,
            sys.link_names,
            sys.link_types,
            P,
            D if D is not None else jnp.zeros((sys.qd_size(),)),
        )
        return PDControllerState(0, q_ref_as_dict, qd_ref_as_dict, P_as_dict, D_as_dict)

    def apply(
        controller_state: PDControllerState, sys: base.System, state: base.State
    ) -> jax.Array:
        taus = jnp.zeros((sys.qd_size()))
        q_ref, qd_ref = jax.tree_map(
            lambda arr: jax.lax.dynamic_index_in_dim(
                arr, controller_state.i, keepdims=False
            ),
            (controller_state.q_ref_as_dict, controller_state.qd_ref_as_dict),
        )

        def f(_, idx_map, idx, name, typ, q_curr, qd_curr):
            nonlocal taus

            if name not in controller_state.q_ref_as_dict:
                return

            p_control_term = jcalc.get_joint_model(typ).p_control_term
            if p_control_term is None:
                raise NotImplementedError(
                    f"Please specify `JointModel.p_control_term` for joint type `{typ}`"
                )
            P_term = p_control_term(q_curr, q_ref[name])
            tau = P_term * controller_state.P_gains[name]

            if name in controller_state.qd_ref_as_dict:
                D_term = (qd_ref[name] - qd_curr) * controller_state.D_gains[name]
                tau += D_term

            taus = taus.at[idx_map["d"](idx)].set(tau)

        sys.scan(
            f,
            "lllqd",
            list(range(sys.num_links())),
            sys.link_names,
            sys.link_types,
            state.q,
            state.qd,
        )

        return controller_state.replace(i=controller_state.i + 1), taus

    return SimpleNamespace(init=init, apply=apply)


def _unroll_dynamics_pd_control(
    sys: base.System,
    q_ref: jax.Array,
    P: jax.Array,
    D: Optional[jax.Array] = None,
    nograv: bool = False,
    sys_q_ref: Optional[base.System] = None,
    initial_sim_state_is_zeros: bool = False,
    clip_taus: Optional[float] = None,
):
    assert q_ref.ndim == 2

    if sys_q_ref is None:
        sys_q_ref = sys

    if nograv:
        sys = sys.replace(gravity=sys.gravity * 0.0)

    if initial_sim_state_is_zeros:
        state = base.State.create(sys)
    else:
        state = _initial_q_is_q_ref(sys, sys_q_ref, q_ref[0])

    controller = _pd_control(P, D)
    cs = controller.init(sys_q_ref, q_ref)

    def step(carry, _):
        state, cs = carry
        cs, taus = controller.apply(cs, sys, state)
        if clip_taus is not None:
            assert clip_taus > 0.0
            taus = jnp.clip(taus, -clip_taus, clip_taus)
        state = dynamics.step(sys, state, taus)
        carry = (state, cs)
        return carry, state

    states = jax.lax.scan(step, (state, cs), None, length=q_ref.shape[0])[1]
    return states


def _initial_q_is_q_ref(sys: base.System, sys_q_ref: base.System, q_ref):
    # you can not preallocate q using zeros because of quaternions..
    q = base.State.create(sys).q

    sys_q_map = sys.idx_map("q")

    def f(_, __, name, q_ref_link):
        nonlocal q
        q = q.at[sys_q_map[name]].set(q_ref_link)

    sys_q_ref.scan(f, "lq", sys_q_ref.link_names, q_ref)

    return base.State.create(sys, q=q)
