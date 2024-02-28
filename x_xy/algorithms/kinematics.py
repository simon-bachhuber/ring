from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import jaxopt
from jaxopt._src.base import Solver

from x_xy import algebra

from .. import base
from .. import maths
from .jcalc import _joint_types
from .jcalc import jcalc_transform


def forward_kinematics_transforms(
    sys: base.System, q: jax.Array
) -> Tuple[base.Transform, base.System]:
    """Perform forward kinematics in system.

    Returns:
        - Transforms from base to links. Transforms first axis is (n_links,).
        - Updated system object with updated `transform2` and `transform` fields.
    """

    eps_to_l = {-1: base.Transform.zero()}

    def update_eps_to_l(_, __, q, link, link_idx, parent_idx, joint_type: str):
        transform2 = jcalc_transform(joint_type, q, link.joint_params)
        transform = algebra.transform_mul(transform2, link.transform1)
        link = link.replace(transform=transform, transform2=transform2)
        eps_to_l[link_idx] = algebra.transform_mul(transform, eps_to_l[parent_idx])
        return eps_to_l[link_idx], link

    eps_to_l_trafos, updated_links = sys.scan(
        update_eps_to_l,
        "qllll",
        q,
        sys.links,
        list(range(sys.num_links())),
        sys.link_parents,
        sys.link_types,
    )
    sys = sys.replace(links=updated_links)
    return (eps_to_l_trafos, sys)


def forward_kinematics(
    sys: base.System, state: base.State
) -> Tuple[base.System, base.State]:
    """Perform forward kinematics in system.
    - Updates `transform` and `transform2` in `sys`
    - Updates `x` in `state`
    """
    x, sys = forward_kinematics_transforms(sys, state.q)
    state = state.replace(x=x)
    return sys, state


def inverse_kinematics_endeffector(
    sys: base.System,
    endeffector_link_name: str,
    endeffector_x: base.Transform,
    error_weight_rot: float = 1.0,
    error_weight_pos: float = 1.0,
    q0: Optional[jax.Array] = None,
    random_q0_starts: Optional[int] = None,
    key: Optional[jax.Array] = None,
    custom_joints: dict[str, Callable[[jax.Array], jax.Array]] = {},
    jaxopt_solver: Solver = jaxopt.LBFGS,
    **jaxopt_solver_kwargs,
) -> tuple[jax.Array, jaxopt.OptStep]:
    """Find the minimal coordinates (joint configuration) such that the endeffector
    reaches a desired rotational and positional configuration / state.

    Args:
        sys (base.System): System under consideration.
        endeffector_link_name (str): Link in system which must reach a desired
            pos & rot state.
        endeffector_x (base.Transform): Desired position and rotation state values.
        error_weight_rot (float, optional): Weight of position error term in
            optimized RMSE loss. Defaults to 1.0.
        error_weight_pos (float, optional): Weight of rotational error term in
            optimized RMSE loss. Defaults to 1.0.
        q0 (Optional[jax.Array], optional): Initial minimal coordinates guess.
            Defaults to None.
        random_q0_starts (Optional[int], optional): Number of random initial values
            to try. Defaults to None.
        key (Optional[jax.Array], optional): PRNGKey, only required if
            `random_q0_starts` > 0. Defaults to None.
        custom_joints (dict[str, Callable[[jax.Array], jax.Array]], optional):
            Dictonary that contains for each custom joint type a function that maps from
            [-inf, inf] -> feasible joint value range. Defaults to {}.
            For example: By default, for a hinge joint it uses `maths.wrap_to_pi`.
        jaxopt_solver (Solver, optional): Solver to use. Defaults to jaxopt.LBFGS.

    Raises:
        NotImplementedError: Specific joint has no preprocess function given in
            `custom_joints`; but this is required.

    Returns:
        tuple[jax.Array, jaxopt.OptStep]:
            Minimal coordinates solution, Residual Loss, Optimizer Results
    """
    assert endeffector_x.ndim() == 1, "Use `jax.vmap` for batching"

    if random_q0_starts is not None:
        assert q0 is None, "Either provide `q0` or `random_q0_starts`."
        assert key is not None, "`random_q0_starts` requires `key`"

    if q0 is None:
        if random_q0_starts is None:
            q0 = base.State.create(sys).q
        else:
            q0s = jax.random.normal(key, shape=(random_q0_starts, sys.q_size()))
            qs, values, results = jax.vmap(
                lambda q0: inverse_kinematics_endeffector(
                    sys,
                    endeffector_link_name,
                    endeffector_x,
                    error_weight_rot,
                    error_weight_pos,
                    q0,
                    None,
                    None,
                    custom_joints,
                    jaxopt_solver,
                    **jaxopt_solver_kwargs,
                )
            )(q0s)

            # find result of best q0 initial value
            best_q_index = jnp.argmin(values)
            best_q, best_q_value = jax.tree_map(
                lambda arr: jax.lax.dynamic_index_in_dim(
                    arr, best_q_index, keepdims=False
                ),
                (
                    qs,
                    values,
                ),
            )
            return best_q, best_q_value, results
    else:
        assert len(q0) == sys.q_size()

    def preprocess_q(q: jax.Array) -> jax.Array:
        # preprocess q
        # - normalize quaternions
        # - hinge joints in [-pi, pi]
        q_preproc = []

        def preprocess(_, __, link_type, q):
            inv_kin_preprocess = _joint_types[link_type].inv_kin_preprocess
            # function in custom_joints has priority over JointModel
            if link_type in custom_joints:
                inv_kin_preprocess = custom_joints[link_type]
            if inv_kin_preprocess is None:
                raise NotImplementedError(
                    f"Please specify the custom joint `{link_type}`"
                    " either using the `custom_joints` arguments or using the"
                    " JointModel.inv_kin_preprocess field."
                )
            new_q = inv_kin_preprocess(q)
            q_preproc.append(new_q)

        sys.scan(preprocess, "lq", sys.link_types, q)
        return jnp.concatenate(q_preproc)

    def objective(q: jax.Array) -> jax.Array:
        q = preprocess_q(q)
        xhat = forward_kinematics_transforms(sys, q)[0][
            sys.name_to_idx(endeffector_link_name)
        ]
        error_rot = maths.angle_error(endeffector_x.rot, xhat.rot)
        error_pos = jnp.sqrt(jnp.sum((endeffector_x.pos - xhat.pos) ** 2))
        return error_weight_rot * error_rot + error_weight_pos * error_pos

    solver = jaxopt_solver(objective, **jaxopt_solver_kwargs)
    results = solver.run(q0)
    q_sol = preprocess_q(results.params)
    # stop gradients such that this value can be used for optimizing e.g.
    # parameters in the system object, such as sys.links.joint_params
    q_sol_value = objective(jax.lax.stop_gradient(results.params))
    return q_sol, q_sol_value, results
