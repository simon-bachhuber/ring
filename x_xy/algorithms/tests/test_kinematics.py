import jax
import jax.numpy as jnp
import tree_utils as tu

import x_xy
from x_xy import maths


def SKIP_test_forward_kinematics_transforms():
    sys = x_xy.load_example("test_kinematics")
    q = [
        jnp.array([1, 0, 0, 0, 1, 1, 1.0]),
        jnp.pi / 2,
        jnp.pi / 2,
        jnp.pi / 4,
        jnp.pi / 2,
    ]
    q = list(map(jnp.atleast_1d, q))
    q = jnp.concatenate(q)
    ts, sys = jax.jit(x_xy.algorithms.forward_kinematics_transforms)(sys, q)

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
        sys = x_xy.load_sys_from_str(xml)

        @jax.jit
        def solve(key):
            c1, c2 = jax.random.split(key)
            random_q = _preprocess_q(sys, jax.random.normal(c1, (sys.q_size(),)))
            endeffector_x = jax.jit(x_xy.forward_kinematics)(
                sys, x_xy.State.create(sys, q=random_q)
            )[1].x[sys.name_to_idx("endeffector")]

            q0 = jax.random.normal(c2, (sys.q_size(),))
            _, results = x_xy.inverse_kinematics_endeffector(
                sys,
                "endeffector",
                endeffector_x,
                q0=q0,
            )
            return results.state.value

        for trial in range(5):
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

    x_xy.scan_sys(sys, preprocess, "lq", sys.link_types, q)
    return jnp.concatenate(q_preproc)
