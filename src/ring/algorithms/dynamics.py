from typing import Optional, Tuple
import warnings

import jax
import jax.numpy as jnp

from ring import algebra
from ring import base
from ring import maths
from ring.algorithms import jcalc
from ring.algorithms import kinematics


def inverse_dynamics(sys: base.System, qd: jax.Array, qdd: jax.Array) -> jax.Array:
    """Performs inverse dynamics in the system. Calculates "tau".
    NOTE: Expects `sys` to have updated `transform` and `inertia`.
    """
    gravity = base.Motion.create(vel=sys.gravity)

    vel, acc, fs = {}, {}, {}

    def forward_scan(_, __, link_idx, parent_idx, link_type, qd, qdd, link):
        p_to_l_trafo, it, joint_params = link.transform, link.inertia, link.joint_params

        vJ = jcalc.jcalc_motion(link_type, qd, joint_params)
        aJ = jcalc.jcalc_motion(link_type, qdd, joint_params)

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

    sys.scan(
        forward_scan,
        "lllddl",
        list(range(sys.num_links())),
        sys.link_parents,
        sys.link_types,
        qd,
        qdd,
        sys.links,
    )

    taus = []

    def backwards_scan(_, __, link_idx, parent_idx, link_type, l_to_p_trafo, link):
        tau = jcalc.jcalc_tau(link_type, fs[link_idx], link.joint_params)
        taus.insert(0, tau)
        if parent_idx != -1:
            fs[parent_idx] = fs[parent_idx] + algebra.transform_force(
                l_to_p_trafo, fs[link_idx]
            )

    sys.scan(
        backwards_scan,
        "lllll",
        list(range(sys.num_links())),
        sys.link_parents,
        sys.link_types,
        jax.vmap(algebra.transform_inv)(sys.links.transform),
        sys.links,
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

    batched_its = sys.scan(
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
        joint_params = (sys.links[i]).joint_params
        link_type = sys.link_types[i]
        # limit scope; only pass in params of this joint type
        joint_params = (
            joint_params[link_type]
            if link_type in joint_params
            else joint_params["default"]
        )

        _to_motion = lambda m: m if isinstance(m, base.Motion) else m(joint_params)
        list_motion = [_to_motion(m) for m in jcalc.get_joint_model(link_type).motion]

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

    sys.scan(populate_H, "l", list(range(sys.num_links())), reverse=True)

    H = H + jnp.tril(H, -1).T

    H += jnp.diag(sys.link_armature)

    return H


def _quaternion_spring_force(q_zeropoint, q) -> jax.Array:
    "Computes the angular velocity direction from q to q_zeropoint."
    qrel = maths.quat_mul(q_zeropoint, maths.quat_inv(q))
    axis, angle = maths.quat_to_rot_axis(qrel)
    return axis * angle


def _spring_force(sys: base.System, q: jax.Array):
    q_spring_force = []

    def _calc_spring_force_per_link(_, __, q, zeropoint, typ):
        # cor is (free, p3d) stacked; free is (spherical, p3d) stacked
        if base.System.joint_type_is_free_or_cor(typ):
            quat_force = _quaternion_spring_force(zeropoint[:4], q[:4])
            pos_force = zeropoint[4:] - q[4:]
            q_spring_force_link = jnp.concatenate((quat_force, pos_force))
        elif base.System.joint_type_is_spherical(typ):
            q_spring_force_link = _quaternion_spring_force(zeropoint, q)
        else:
            q_spring_force_link = zeropoint - q
        q_spring_force.append(q_spring_force_link)

    sys.scan(
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
    # mass_mat_inv: jax.Array,
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
        warnings.warn(
            f"You are using `sys.mass_mat_iters`={sys.mass_mat_iters} which is >0. "
            "This feature is currently not fully supported. See the local TODO."
        )
        mass_mat_inv = jnp.diag(jnp.ones((sys.qd_size(),)))
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


def _semi_implicit_euler_integration(
    sys: base.System, state: base.State, taus: jax.Array
) -> base.State:
    qdd, mass_mat_inv = forward_dynamics(sys, state.q, state.qd, taus)
    del mass_mat_inv
    qd_next = state.qd + sys.dt * qdd

    q_next = []

    def q_integrate(_, __, q, qd, typ):
        if sys.joint_type_is_free_or_cor(typ):
            quat_next = _strapdown_integration(q[:4], qd[:3], sys.dt)
            pos_next = q[4:] + qd[3:] * sys.dt
            q_next_i = jnp.concatenate((quat_next, pos_next))
        elif sys.joint_type_is_spherical(typ):
            quat_next = _strapdown_integration(q, qd, sys.dt)
            q_next_i = quat_next
        else:
            q_next_i = q + sys.dt * qd
        q_next.append(q_next_i)

    # uses already `qd_next` because semi-implicit
    sys.scan(q_integrate, "qdl", state.q, qd_next, sys.link_types)
    q_next = jnp.concatenate(q_next)

    state = state.replace(q=q_next, qd=qd_next)
    return state


_integration_methods = {
    "semi_implicit_euler": _semi_implicit_euler_integration,
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
    "Steps the dynamics. Returns the state of next timestep."
    assert sys.q_size() == state.q.size
    if taus is None:
        taus = jnp.zeros_like(state.qd)
    assert sys.qd_size() == state.qd.size == taus.size
    assert (
        sys.integration_method.lower() == "semi_implicit_euler"
    ), "Currently, nothing else then `semi_implicit_euler` implemented."

    sys = sys.replace(dt=sys.dt / n_substeps)

    for _ in range(n_substeps):
        # update kinematics before stepping; this means that the `x` in `state`
        # will lag one step behind but otherwise we would have to return
        # the system object which would be awkward
        sys, state = kinematics.forward_kinematics(sys, state)
        state = _integration_methods[sys.integration_method.lower()](sys, state, taus)

    return state


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
