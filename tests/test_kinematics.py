import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import tree_utils as tu

import ring
from ring import base
from ring import maths
from ring.algorithms import jcalc
from ring.algorithms import kinematics
from ring.sim2real.sim2real import _checks_time_series_of_xs


def test_forward_kinematics_transforms():
    sys = ring.io.load_example("test_kinematics")
    q = [
        jnp.array([1, 0, 0, 0, 1, 1, 1.0]),
        jnp.pi / 2,
        jnp.pi / 2,
        jnp.pi / 4,
        jnp.pi / 2,
    ]
    q = list(map(jnp.atleast_1d, q))
    q = jnp.concatenate(q)
    ts, sys = jax.jit(kinematics.forward_kinematics_transforms)(sys, q)

    # position ok
    assert tu.tree_close(ts.take(4).pos, jnp.array([2.0, 2, 1]))

    # orientation ok
    q2_eps = sys.links.transform2.take(2).rot
    q3_2 = sys.links.transform2.take(3).rot
    q4_2 = sys.links.transform.take(4).rot
    assert tu.tree_close(maths.quat_mul(q3_2, q2_eps), ts.take(3).rot)
    assert tu.tree_close(maths.quat_mul(q4_2, q2_eps), ts.take(4).rot)


def test_inv_kinematics_endeffector():
    xml1 = """
<x_xy model="spherical">
    <worldbody>
        <body name="s1" joint="spherical">
        <geom type="box" pos="0.5 0 0" dim="1 0.2 0.2" mass="0"/>
        <body name="s2" joint="spherical" pos="1 0 0">
            <geom type="box" pos="0.5 0 0" dim="1 0.2 0.2" mass="0"/>
                <body name="endeffector" joint="frozen" pos="1 0 0">
                    <geom type="sphere" mass="0" dim="0.1" color="target"/>
                </body>
            </body>
        </body>
    </worldbody>
</x_xy>
"""
    xml2 = """
<x_xy model="2D">
    <worldbody>
        <body name="s1" joint="ry">
            <body name="s2" joint="px">
                <body name="endeffector" joint="ry">
                </body>
            </body>
        </body>
    </worldbody>
</x_xy>
"""
    key = jax.random.PRNGKey(222)
    for xml in [xml1, xml2]:
        sys = ring.io.load_sys_from_str(xml)

        @jax.jit
        def solve(key):
            c1, c2 = jax.random.split(key)
            random_q = _preprocess_q(sys, jax.random.normal(c1, (sys.q_size(),)))
            endeffector_x = jax.jit(kinematics.forward_kinematics)(
                sys, ring.State.create(sys, q=random_q)
            )[1].x[sys.name_to_idx("endeffector")]

            q0 = jax.random.normal(c2, (sys.q_size(),))
            _, value, _ = kinematics.inverse_kinematics_endeffector(
                sys,
                "endeffector",
                endeffector_x,
                q0=q0,
                jaxopt_solver=jaxopt.GradientDescent,
                maxiter=5000,
                maxls=3,
            )
            return value

        for trial in range(20):
            print(sys.model_name, trial)
            key, consume = jax.random.split(key)
            assert solve(consume) < 1e-3


def _preprocess_q(sys, q: jax.Array) -> jax.Array:
    # preprocess q
    # - normalize quaternions
    # - hinge joints in [-pi, pi]
    q_preproc = []

    def preprocess(_, __, link_type, q):
        if link_type in ["free", "cor", "spherical"]:
            new_q = q.at[:4].set(maths.safe_normalize(q[:4]))
        elif link_type in ["rx", "ry", "rz", "saddle"]:
            new_q = maths.wrap_to_pi(q)
        elif link_type in ["frozen", "p3d", "px", "py", "pz"]:
            new_q = q
        else:
            raise NotImplementedError
        q_preproc.append(new_q)

    sys.scan(preprocess, "lq", sys.link_types, q)
    return jnp.concatenate(q_preproc)


_str2idx = {"x": 0, "y": 1, "z": 2}


def _inv_kin_rxyz_factory(xyz: str):
    def _inv_kin_rxyz(x: base.Transform, _) -> jax.Array:
        angles = maths.quat_to_euler(x.rot)
        idx = _str2idx[xyz]
        proj_angles = jnp.zeros((3,)).at[idx].set(angles[idx])
        rot = maths.euler_to_quat(proj_angles)
        return base.Transform.create(rot=rot)

    return _inv_kin_rxyz


def _project_transform_to_feasible_pxyz_factory(xyz: str):
    def _project_transform_to_feasible_pxyz(x: base.Transform, _) -> base.Transform:
        idx = _str2idx[xyz]
        pos = jnp.zeros((3,)).at[idx].set(x.pos[idx])
        return base.Transform.create(pos=pos)

    return _project_transform_to_feasible_pxyz


def project_xs(sys: base.System, transform2: base.Transform) -> base.Transform:
    """Project transforms into the physically feasible subspace as defined by the
    joints in the system."""
    _checks_time_series_of_xs(sys, transform2)

    @jax.vmap
    def _project_xs(transform2):
        def f(_, __, i: int, link_type: str, link):
            t = transform2[i]
            joint_params = link.joint_params
            # limit scope
            joint_params = (
                joint_params[link_type]
                if link_type in joint_params
                else joint_params["default"]
            )

            project_transform_to_feasible = jcalc.get_joint_model(
                link_type
            ).project_transform_to_feasible
            if project_transform_to_feasible is None:
                raise NotImplementedError(
                    "Please specify JointModel.project_transform_to_feasible"
                    f" for joint type `{link_type}`."
                )
            return project_transform_to_feasible(t, joint_params)

        return sys.scan(
            f,
            "lll",
            list(range(sys.num_links())),
            sys.link_types,
            sys.links,
        )

    return _project_xs(transform2)


def TODO_test_eq_inverse_kinematics_and_project_xs():
    pass


def test_inverse_kinematics_forward_kinematics():
    sys = ring.io.load_example("test_three_seg_seg2")

    for seed in range(5):
        print(seed)
        state = base.State.create(sys, key=jax.random.PRNGKey(seed))
        _, state = kinematics.forward_kinematics(sys, state)
        q = state.q
        state = kinematics.inverse_kinematics(sys, state)
        np.testing.assert_allclose(q, state.q, rtol=1e-6, atol=1e-8)
