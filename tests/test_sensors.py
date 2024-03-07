import jax
import jax.numpy as jnp
import numpy as np
import ring
from ring.algorithms.sensors import rescale_natural_units


def _simulate_imus(qd: float, no_grav: bool = False):
    T = 1.0
    sys = ring.io.load_example("test_sensors")
    assert sys.model_name == "test_sensors"

    if no_grav:
        sys = sys.replace(gravity=sys.gravity * 0.0)

    q = jnp.array([0.0])
    qd = jnp.array(
        [
            qd,
        ]
    )
    state = ring.base.State.create(sys, q, qd)
    xs = []
    tau = jnp.zeros_like(state.qd)
    for _ in range(int(T / sys.dt)):
        state = jax.jit(ring.step)(sys, state, tau)
        xs.append(state.x)
    xs = xs[0].batch(*xs[1:])

    imu1 = ring.algorithms.imu(
        xs.take(sys.name_to_idx("imu1"), axis=1), sys.gravity, sys.dt
    )
    imu2 = ring.algorithms.imu(
        xs.take(sys.name_to_idx("imu2"), axis=1), sys.gravity, sys.dt
    )
    return imu1, imu2


def test_imu():
    # test sensors for continuously rotating system with
    # no energy losses and no gravity
    omega = 2.0
    imu1, imu2 = _simulate_imus(omega, True)

    repeat = lambda arr: jnp.repeat(arr[None], imu1["acc"].shape[0], axis=0)

    np.testing.assert_allclose(
        imu1["gyr"], repeat(jnp.array([0, omega, 0])), 1e-4, 1e-4
    )
    np.testing.assert_allclose(
        imu2["gyr"], repeat(jnp.array([omega, 0, 0])), 1e-4, 1e-4
    )

    # exclude first and final value, because those acc measurements are just copied
    # values but their rotation matrices are not copied, thus the measurements at
    # those two timepoints don't have to be perfect
    np.testing.assert_allclose(
        imu1["acc"][1:-1], repeat(jnp.array([-(omega**2), 0, 0]))[1:-1], 5e-3, 5e-3
    )
    # r-vector is 2.0
    np.testing.assert_allclose(
        imu2["acc"][1:-1], repeat(jnp.array([0, omega**2 * 2, 0]))[1:-1], 5e-3, 1e-2
    )

    # test sensors for static system
    # only gravity should be measured

    imu1, imu2 = _simulate_imus(0.0)
    np.testing.assert_allclose(imu1["gyr"], repeat(jnp.zeros((3,))), atol=1e-4)
    np.testing.assert_allclose(imu2["gyr"], repeat(jnp.zeros((3,))), atol=1e-4)
    np.testing.assert_allclose(imu1["acc"], repeat(jnp.array([-9.81, 0, 0])), atol=2e-3)
    np.testing.assert_allclose(imu2["acc"], repeat(jnp.array([0, 9.81, 0])), atol=4e-3)


def test_rel_pose():
    sys = ring.io.load_example("test_sensors")
    qs = ring.maths.quat_random(
        jax.random.PRNGKey(
            1,
        ),
        (3,),
    )
    xs = ring.base.Transform.create(rot=qs)
    y = ring.algorithms.rel_pose(sys, xs)

    assert "hinge" not in y

    qrel = lambda q1, q2: ring.maths.quat_mul(q1, ring.maths.quat_inv(q2))
    np.testing.assert_array_equal(y["imu1"], qrel(qs[0], qs[1]))
    np.testing.assert_array_equal(y["imu2"], qrel(qs[0], qs[2]))

    sys_different_anchor = r"""
    <x_xy model="test_sensors">
    <options gravity="0 0 9.81" dt="0.01"/>
    <defaults>
        <geom edge_color="black" color="white"/>
    </defaults>
    <worldbody>
        <body name="hinge" pos="0 0 0" euler="0 90 0" joint="ry">
            <geom type="box" mass="10" pos="0.25 0 0" dim="0.5 0.25 0.2"/>
            <body name="imu2" pos="1 0 0" joint="frozen">
                <body name="imu1" pos="2 0 0" euler="0 0 90" joint="frozen"></body>
            </body>
        </body>
    </worldbody>
    </x_xy>
    """

    y = ring.algorithms.rel_pose(
        ring.io.load_sys_from_str(sys_different_anchor), xs, sys
    )
    assert "hinge" not in y
    np.testing.assert_array_equal(y["imu2"], qrel(qs[0], qs[2]))
    np.testing.assert_array_equal(y["imu1"], qrel(qs[2], qs[1]))


def test_natural_units():
    sys = ring.io.load_example("test_three_seg_seg2")
    X, y = ring.RCMG(
        sys,
        add_X_imus=True,
        add_X_imus_kwargs=dict(natural_units=False),
    ).to_list()[0]
    X_nat, y_nat = ring.RCMG(
        sys,
        add_X_imus=True,
        add_X_imus_kwargs=dict(natural_units=True),
    ).to_list()[0]

    imu_name = "seg1"
    imu, imu_nat = X[imu_name], X_nat[imu_name]

    np.testing.assert_allclose(imu["gyr"], imu_nat["gyr"] * np.pi, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(imu["acc"], imu_nat["acc"] * 9.81, atol=1e-6, rtol=1e-6)

    # just test that this works
    {key: rescale_natural_units(val) for key, val in X.items()}
