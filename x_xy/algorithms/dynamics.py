from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from flax import struct

from x_xy.algorithms import forward_kinematics_transforms
from x_xy.base import Force, Motion, State, System


def inverse_dynamics(sys: System, qd: jax.Array, qdd: jax.Array) -> jax.Array:
    # TODO
    # should be -9.81 to pass tests; Roy uses -9.81
    # however vispy has z-axis upwards, then +9.81 better
    gravity = Motion.create(vel=sys.gravity)

    vJ = sys.links.joint.motion.vmap().__mul__(qd)
    aJ = sys.links.joint.motion.vmap().__mul__(qdd)
    del qd, qdd

    vel = Motion.zero((sys.N,))
    acc = Motion.zero((sys.N,))
    fs = Force.zero((sys.N,))

    def body_fn(i: int, val):
        vel, acc, fs = val
        p = sys.parent[i]
        t = sys.links.transform.take(i)

        def parent_is_root():
            v = vJ.take(i)
            a = t.do(gravity) + aJ.take(i)
            return v, a

        def parent_not_root():
            v = t.do(vel.take(p)) + vJ.take(i)
            a = t.do(acc.take(p)) + aJ.take(i) + v.cross(vJ.take(i))
            return v, a

        v, a = jax.lax.cond(p == -1, parent_is_root, parent_not_root)
        vel = vel.index_set(i, v)
        acc = acc.index_set(i, a)
        inertia_i = sys.links.inertia.take(i)
        f = inertia_i.mul(a) + v.cross(inertia_i.mul(v))
        fs = fs.index_set(i, f)
        return vel, acc, fs

    vel, acc, fs, *_ = jax.lax.fori_loop(0, sys.N, body_fn, (vel, acc, fs))

    def scan_fn(carry, _):
        fs, i = carry
        p = sys.parent[i]
        tau = sys.links.joint.motion.take(i).dot(fs.take(i))

        def parent_is_root(fs):
            return fs

        def parent_not_root(fs):
            return fs.index_sum(p, sys.links.transform.take(i).inv().do(fs.take(i)))

        fs = jax.lax.cond(p == -1, parent_is_root, parent_not_root, fs)
        return (fs, i - 1), tau

    (fs, stop), taus = jax.lax.scan(scan_fn, (fs, sys.N - 1), None, length=sys.N)

    # now tau is in revered order; because we scanned backwards
    taus = jnp.flip(taus)

    return taus, stop  # stop must be -1


def _compute_mass_matrix(sys: System) -> jax.Array:
    t = sys.links.transform
    s = sys.links.joint.motion

    def body_fn(i, val):
        # jax does not support negative increments; naive workaround
        i = sys.N - 1 - i

        H, inertia = val
        p = sys.parent[i]

        # Part 2 | Figure 7 | Line 7
        def parent_is_root(inertia, p):
            return inertia

        def parent_not_root(inertia, p):
            return inertia.index_sum(p, t.take(i).inv().do(inertia.take(i)))

        inertia = jax.lax.cond(p == -1, parent_is_root, parent_not_root, inertia, p)
        del p

        f = inertia.take(i).mul(s.take(i))
        H = H.at[i, i].set(f.dot(s.take(i)))

        def from_link_directly_to_root(val):
            j, H, f = val
            f = t.take(j).inv().do(f)
            j = sys.parent[j]
            H = H.at[i, j].set(f.dot(s.take(j)))
            return (j, H, f)

        def cond_fn_parent_not_root(val):
            return sys.parent[val[0]] != -1

        _, H, _ = jax.lax.while_loop(
            cond_fn_parent_not_root, from_link_directly_to_root, (i, H, f)
        )

        return H, inertia

    H = jnp.zeros((sys.N, sys.N))
    H, _ = jax.lax.fori_loop(0, sys.N, body_fn, (H, sys.links.inertia))
    H = H + jnp.tril(H, -1).T

    # TODO
    # understand this; brax/mass.py line 80
    H = H + jnp.diag(sys.links.joint.squeeze_1d().armature)

    return H


def _forward_dynamics(
    sys: System, q, qd, taus, mode: int, timestep: Optional[float] = None
) -> jax.Array:
    joint = sys.links.joint.squeeze_1d()

    C, _ = inverse_dynamics(sys, qd, jnp.zeros_like(q))
    H = _compute_mass_matrix(sys)
    spring_force = joint.stiffness * (joint.zero_position - q) - joint.damping * qd
    qf_smooth = spring_force + taus - C

    if mode != 0:
        assert timestep is not None, "Only in mode `0` is a timestep optional"

    if mode == 0:
        return jax.scipy.linalg.solve(H, qf_smooth, assume_a="pos")
    elif mode == 1:
        H_damp = H + timestep * jnp.diag(joint.damping)
        return jax.scipy.linalg.solve(H_damp, qf_smooth, assume_a="pos")
    elif mode == 2:
        H_inv = jax.scipy.linalg.solve(H, jnp.eye(sys.N), assume_a="pos")
        H_inv_damp = H_inv - H_inv @ (jnp.diag(joint.damping) * timestep) @ H_inv
        return H_inv_damp @ qf_smooth
    else:
        raise NotImplementedError


def forward_dynamics(sys: System, q, qd, taus) -> jax.Array:
    return _forward_dynamics(sys, q, qd, taus, 0)


def _explicit_euler(rhs, t, x, dt):
    dx = rhs(t, x)
    return x + dx * dt


def _runge_kutta(rhs, t, x, dt):
    h = dt
    k1 = rhs(t, x)
    k2 = rhs(t + h / 2, x + k1 * (h / 2))
    k3 = rhs(t + h / 2, x + k2 * (h / 2))
    k4 = rhs(t + h, x + k3 * h)
    dx = (k1 + k2 * 2 + k3 * 2 + k4) * (1 / 6)
    return x + dx * dt


@struct.dataclass
class SolverParams:
    mode: int


def _solve_explicit(
    sys, state, taus, timestep, solver_params: SolverParams, integrator_fn
):
    def rhs(t, x):
        q, qd = x[: sys.N], x[sys.N :]
        qdd = _forward_dynamics(sys, q, qd, taus, solver_params.mode, timestep)
        return jnp.hstack((qd, qdd))

    x = jnp.hstack((state.q, state.qd))
    x_next = integrator_fn(rhs, 0.0, x, timestep)
    q_next, qd_next = x_next[: sys.N], x_next[sys.N :]
    return q_next, qd_next


def _semi_implicit_euler(sys, state, taus, timestep, solver_params: SolverParams):
    qdd = _forward_dynamics(sys, state.q, state.qd, taus, solver_params.mode, timestep)
    qd_next = state.qd + timestep * qdd
    q_next = state.q + timestep * qd_next
    return q_next, qd_next


_solver_params_defaults = {
    "SIE": SolverParams(0),  # mode 2?
    "EE": SolverParams(0),
    "RK4": SolverParams(0),
}


_solvers = {
    "SIE": _semi_implicit_euler,
    "EE": lambda *args: _solve_explicit(*args, _explicit_euler),
    "RK4": lambda *args: _solve_explicit(*args, _runge_kutta),
}


@partial(jax.jit, static_argnums=(3, 5))
def simulation_step(
    sys: System,
    state: State,
    taus: jax.Array,
    integrator="sie",
    timestep: float = 0.01,
    solver_params: Optional[SolverParams] = None,
):
    if solver_params is None:
        solver_params = _solver_params_defaults[integrator.upper()]

    # update all transforms
    sys = update_link_transform(sys, state.q)

    # compute maximal coordinates
    # TODO
    # maximal coordinates will lag one frame behind
    x_max_cord = forward_kinematics(sys)
    state = state.replace(x=x_max_cord)

    q_next, qd_next = _solvers[integrator.upper()](
        sys, state, taus, timestep, solver_params
    )

    return state.replace(q=q_next, qd=qd_next)
