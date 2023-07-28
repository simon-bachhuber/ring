from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from x_xy import algebra, algorithms, base, maths, scan


def inverse_dynamics(sys: base.System, qd: jax.Array, qdd: jax.Array) -> jax.Array:
    """Performs inverse dynamics in the system. Calculates "tau".
    NOTE: Expects `sys` to have updated `transform` and `inertia`.
    """
    gravity = base.Motion.create(vel=sys.gravity)

    vel, acc, fs = {}, {}, {}

    def forward_scan(_, __, link_idx, parent_idx, link_type, qd, qdd, p_to_l_trafo, it):
        vJ, aJ = algorithms.jcalc_motion(link_type, qd), algorithms.jcalc_motion(
            link_type, qdd
        )

        t = lambda m: algebra.transform_motion(p_to_l_trafo, m)

        if parent_idx == -1:
            v = vJ
            a = t(gravity) + aJ
        else:
            v = vJ + t(vel[parent_idx])
            a = t(acc[parent_idx]) + aJ + algebra.motion_cross(v, vJ)

        vel[link_idx], acc[link_idx] = v, a
        f = algebra.inertia_mul_motion(it, a) + algebra.motion_cross_star(
            v, algebra.inertia_mul_motion(it, v)
        )
        fs[link_idx] = f

    scan.tree(
        sys,
        forward_scan,
        "lllddll",
        list(range(sys.num_links())),
        sys.link_parents,
        sys.link_types,
        qd,
        qdd,
        sys.links.transform,
        sys.links.inertia,
    )

    taus = []

    def backwards_scan(_, __, link_idx, parent_idx, link_type, l_to_p_trafo):
        tau = algorithms.jcalc_tau(link_type, fs[link_idx])
        taus.insert(0, tau)
        if parent_idx != -1:
            fs[parent_idx] = fs[parent_idx] + algebra.transform_force(
                l_to_p_trafo, fs[link_idx]
            )

    scan.tree(
        sys,
        backwards_scan,
        "llll",
        list(range(sys.num_links())),
        sys.link_parents,
        sys.link_types,
        jax.vmap(algebra.transform_inv)(sys.links.transform),
        reverse=True,
    )

    return jnp.concatenate(taus)


def compute_mass_matrix(sys: base.System) -> jax.Array:
    """Computes the mass matrix of the system using the `composite-rigid-body`
    algorithm."""

    # STEP 1: Accumulate inertias inwards
    # We will stay in spatial mode in this step
    l_to_p = jax.vmap(algebra.transform_inv)(sys.links.transform)
    its = [sys.links.inertia[link_idx] for link_idx in range(sys.num_links())]

    def accumulate_inertias(_, __, i, p):
        nonlocal its
        if p != -1:
            its[p] += algebra.transform_inertia(l_to_p[i], its[i])
        return its[i]

    batched_its = scan.tree(
        sys,
        accumulate_inertias,
        "ll",
        list(range(sys.num_links())),
        sys.link_parents,
        reverse=True,
    )

    # express inertias as matrices (in a vectorized way)
    @jax.vmap
    def to_matrix(obj):
        return obj.as_matrix()

    I_mat = to_matrix(batched_its)
    del its, batched_its

    # STEP 2: Populate mass matrix
    # Now we go into matrix mode

    def _jcalc_motion_matrix(i: int):
        list_motion = algorithms.jcalc._joint_types[sys.link_types[i]].motion
        if len(list_motion) == 0:
            # joint is frozen
            return None
        stacked_motion = list_motion[0].batch(*list_motion[1:])
        return to_matrix(stacked_motion)

    S = [_jcalc_motion_matrix(i) for i in range(sys.num_links())]

    H = jnp.zeros((sys.qd_size(), sys.qd_size()))

    def populate_H(_, idx_map, i):
        nonlocal H

        # frozen joint type
        if S[i] is None:
            return

        f = (I_mat[i] @ (S[i].T)).T
        idxs_i = idx_map["d"](i)
        H_ii = f @ (S[i].T)

        # set upper diagonal entries to zero
        # they will be filled later automatically
        H_ii_lower = jnp.tril(H_ii)
        H = H.at[idxs_i, idxs_i].set(H_ii_lower)

        j = i
        parent = lambda i: sys.link_parents[i]
        while parent(j) != -1:

            @jax.vmap
            def transform_force(f_arr):
                spatial_f = base.Force(f_arr[:3], f_arr[3:])
                spatial_f_in_p = algebra.transform_force(l_to_p[j], spatial_f)
                return spatial_f_in_p.as_matrix()

            # transforms force into parent frame
            f = transform_force(f)

            j = parent(j)
            if S[j] is None:
                continue

            H_ij = f @ (S[j].T)
            idxs_j = idx_map["d"](j)
            H = H.at[idxs_i, idxs_j].set(H_ij)

    scan.tree(sys, populate_H, "l", list(range(sys.num_links())), reverse=True)

    H = H + jnp.tril(H, -1).T

    H += jnp.diag(sys.link_armature)

    return H


def _quaternion_spring_force(q_zeropoint, q) -> jax.Array:
    "Computes the angular velocity direction from q to q_zeropoint."
    qrel = maths.quat_mul(q_zeropoint, maths.quat_inv(q))
    return maths.quat_to_rotvec(qrel)


def _spring_force(sys: base.System, q: jax.Array):
    q_spring_force = []

    def _calc_spring_force_per_link(_, __, q, zeropoint, typ):
        if typ == "free":
            quat_force = _quaternion_spring_force(zeropoint[:4], q[:4])
            pos_force = zeropoint[4:] - q[4:]
            q_spring_force_link = jnp.concatenate((quat_force, pos_force))
        elif typ == "spherical":
            q_spring_force_link = _quaternion_spring_force(zeropoint, q)
        else:
            q_spring_force_link = zeropoint - q
        q_spring_force.append(q_spring_force_link)

    scan.tree(
        sys,
        _calc_spring_force_per_link,
        "qql",
        q,
        sys.link_spring_zeropoint,
        sys.link_types,
    )
    return jnp.concatenate(q_spring_force)


def forward_dynamics(
    sys: base.System,
    q: jax.Array,
    qd: jax.Array,
    tau: jax.Array,
    mass_mat_inv: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    C = inverse_dynamics(sys, qd, jnp.zeros_like(qd))
    mass_matrix = compute_mass_matrix(sys)

    spring_force = -sys.link_damping * qd + sys.link_spring_stiffness * _spring_force(
        sys, q
    )
    qf_smooth = tau - C + spring_force

    if sys.mass_mat_iters == 0:
        eye = jnp.eye(sys.qd_size())

        # trick from brax / mujoco aka "integrate joint damping implicitly"
        mass_matrix += jnp.diag(sys.link_damping) * sys.dt

        # make cholesky decomposition not sometimes fail
        # see: https://github.com/google/jax/issues/16149
        mass_matrix += eye * 1e-6

        mass_mat_inv = jax.scipy.linalg.solve(mass_matrix, eye, assume_a="pos")
    else:
        mass_mat_inv = _inv_approximate(mass_matrix, mass_mat_inv, sys.mass_mat_iters)

    return mass_mat_inv @ qf_smooth, mass_mat_inv


def _strapdown_integration(
    q: base.Quaternion, dang: jax.Array, dt: float
) -> base.Quaternion:
    dang_norm = jnp.linalg.norm(dang) + 1e-8
    axis = dang / dang_norm
    angle = dang_norm * dt
    q = maths.quat_mul(maths.quat_rot_axis(axis, angle), q)
    # Roy book says that one should re-normalize after every quaternion step
    return q / jnp.linalg.norm(q)


def _rk4(rhs, x, dt):
    h = dt
    k1 = rhs(x)
    k2 = rhs(x + k1 * (h / 2))
    k3 = rhs(x + k2 * (h / 2))
    k4 = rhs(x + k3 * h)
    dx = (k1 + k2 * 2 + k3 * 2 + k4) * (1 / 6)
    return x + dx * dt


def _runge_kutta_4_integration(
    sys: base.System, state: base.State, taus: jax.Array
) -> base.State:
    def integrate_q_qd(q, qd, qdd, dt):
        q_next = []

        def q_integrate(_, __, q, qd, typ):
            if typ == "free":
                quat_next = _strapdown_integration(q[:4], qd[:3], dt)
                pos_next = q[4:] + qd[4:] * dt
                q_next_i = jnp.concatenate((quat_next, pos_next))
            else:
                q_next_i = q + dt * qd
            q_next.append(q_next_i)

        scan.tree(sys, q_integrate, "qdl", q, qd, sys.link_types)
        q_next = jnp.concatenate(q_next)
        return q_next, qd + qdd * dt

    for_dyn = lambda sys, q, qd: forward_dynamics(sys, q, qd, taus, state.mass_mat_inv)

    # k1
    qd_k1 = state.qd
    qdd_k1, mass_mat_inv_next = for_dyn(sys, state.q, state.qd)

    # k2
    q_k2, qd_k2 = integrate_q_qd(state.q, qd_k1, qdd_k1, sys.dt * 0.5)
    _, sys = algorithms.kinematics.forward_kinematics_transforms(sys, q_k2)
    qdd_k2, _ = for_dyn(sys, q_k2, qd_k2)

    # k3
    q_k3, qd_k3 = integrate_q_qd(state.q, qd_k2, qdd_k2, sys.dt * 0.5)
    _, sys = algorithms.kinematics.forward_kinematics_transforms(sys, q_k3)
    qdd_k3, _ = for_dyn(sys, q_k3, qd_k3)

    # k4
    q_k4, qd_k4 = integrate_q_qd(state.q, qd_k3, qdd_k3, sys.dt)
    _, sys = algorithms.kinematics.forward_kinematics_transforms(sys, q_k4)
    qdd_k4, _ = for_dyn(sys, q_k4, qd_k4)

    # average over k`s
    reduce = lambda k1, k2, k3, k4: (k1 + 2 * k2 + 2 * k3 + k4) * (1 / 6)
    mqd, mqdd = reduce(qd_k1, qd_k2, qd_k3, qd_k4), reduce(
        qdd_k1, qdd_k2, qdd_k3, qdd_k4
    )

    # forward integrate with averaged delta
    q, qd = integrate_q_qd(state.q, mqd, mqdd, sys.dt)

    # update mass matrix inverse
    state = state.replace(q=q, qd=qd, mass_mat_inv=mass_mat_inv_next)
    return state


def _runge_kutta_2_integration(
    sys: base.System, state: base.State, taus: jax.Array
) -> base.State:
    def integrate_q_qd(q, qd, qdd, dt):
        q_next = []

        def q_integrate(_, __, q, qd, typ):
            if typ == "free":
                quat_next = _strapdown_integration(q[:4], qd[:3], dt)
                pos_next = q[4:] + qd[4:] * dt
                q_next_i = jnp.concatenate((quat_next, pos_next))
            else:
                q_next_i = q + dt * qd
            q_next.append(q_next_i)

        scan.tree(sys, q_integrate, "qdl", q, qd, sys.link_types)
        q_next = jnp.concatenate(q_next)
        return q_next, qd + qdd * dt

    for_dyn = lambda sys, q, qd: forward_dynamics(sys, q, qd, taus, state.mass_mat_inv)

    # k1
    qd_k1 = state.qd
    qdd_k1, mass_mat_inv_next = for_dyn(sys, state.q, state.qd)

    # k2
    q_k2, qd_k2 = integrate_q_qd(state.q, qd_k1, qdd_k1, sys.dt)
    _, sys = algorithms.kinematics.forward_kinematics_transforms(sys, q_k2)
    qdd_k2, _ = for_dyn(sys, q_k2, qd_k2)

    # average over k`s
    reduce = lambda k1, k2: (k1 + k2) * (1 / 2)
    mqd, mqdd = reduce(qd_k1, qd_k2), reduce(qdd_k1, qdd_k2)

    # forward integrate with averaged delta
    q, qd = integrate_q_qd(state.q, mqd, mqdd, sys.dt)

    # update mass matrix inverse
    state = state.replace(q=q, qd=qd, mass_mat_inv=mass_mat_inv_next)
    return state


def _semi_implicit_euler_integration(
    sys: base.System, state: base.State, taus: jax.Array
) -> base.State:
    qdd, mass_mat_inv = forward_dynamics(
        sys, state.q, state.qd, taus, state.mass_mat_inv
    )
    qd_next = state.qd + sys.dt * qdd

    q_next = []

    def q_integrate(_, __, q, qd, typ):
        if typ == "free":
            quat_next = _strapdown_integration(q[:4], qd[:3], sys.dt)
            pos_next = q[4:] + qd[3:] * sys.dt
            q_next_i = jnp.concatenate((quat_next, pos_next))
        elif typ == "spherical":
            quat_next = _strapdown_integration(q, qd, sys.dt)
            q_next_i = quat_next
        else:
            q_next_i = q + sys.dt * qd
        q_next.append(q_next_i)

    # uses already `qd_next` because semi-implicit
    scan.tree(sys, q_integrate, "qdl", state.q, qd_next, sys.link_types)
    q_next = jnp.concatenate(q_next)

    state = state.replace(q=q_next, qd=qd_next, mass_mat_inv=mass_mat_inv)
    return state


_integration_methods = {
    "semi_implicit_euler": _semi_implicit_euler_integration,
    "runge_kutta_4": _runge_kutta_4_integration,
    "runge_kutta_2": _runge_kutta_2_integration,
}


def kinetic_energy(sys: base.System, qd: jax.Array):
    H = compute_mass_matrix(sys)
    return 0.5 * qd @ H @ qd


def step(
    sys: base.System,
    state: base.State,
    taus: Optional[jax.Array] = None,
    n_substeps: int = 1,
) -> base.State:
    assert sys.q_size() == state.q.size
    if taus is None:
        taus = jnp.zeros_like(state.qd)
    assert sys.qd_size() == state.qd.size == taus.size
    assert (
        sys.integration_method.lower() == "semi_implicit_euler"
    ), "Currently Runge-Kutta methods are broken."

    sys = sys.replace(dt=sys.dt / n_substeps)

    def substep(state, _):
        nonlocal sys
        # update kinematics before stepping; this means that the `x` in `state`
        # will lag one step behind but otherwise we would have to return
        # the system object which would be awkward
        sys_updated, state = algorithms.kinematics.forward_kinematics(sys, state)
        state = _integration_methods[sys.integration_method.lower()](
            sys_updated, state, taus
        )
        return state, _

    return jax.lax.scan(substep, state, None, length=n_substeps)[0]


def _inv_approximate(a: jax.Array, a_inv: jax.Array, num_iter: int = 10) -> jax.Array:
    """Use Newton-Schulz iteration to solve ``A^-1``.

    Args:
      a: 2D array to invert
      a_inv: approximate solution to A^-1
      num_iter: number of iterations

    Returns:
      A^-1 inverted matrix
    """

    def body_fn(carry, _):
        a_inv, r, err = carry
        a_inv_next = a_inv @ (jnp.eye(a.shape[0]) + r)
        r_next = jnp.eye(a.shape[0]) - a @ a_inv_next
        err_next = jnp.linalg.norm(r_next)
        a_inv_next = jnp.where(err_next < err, a_inv_next, a_inv)
        return (a_inv_next, r_next, err_next), None

    # ensure ||I - X0 @ A|| < 1, in order to guarantee convergence
    r0 = jnp.eye(a.shape[0]) - a @ a_inv
    a_inv = jnp.where(jnp.linalg.norm(r0) > 1, 0.5 * a.T / jnp.trace(a @ a.T), a_inv)
    (a_inv, _, _), _ = jax.lax.scan(body_fn, (a_inv, r0, 1.0), None, num_iter)

    return a_inv
