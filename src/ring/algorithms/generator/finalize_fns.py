from typing import Optional
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import tree_utils

from ring import base
from ring import utils
from ring.algorithms import sensors
from ring.algorithms.generator import pd_control
from ring.algorithms.generator import types


class FinalizeFn:
    def __init__(self, finalize_fn: types.FINALIZE_FN):
        self.finalize_fn = finalize_fn

    def __call__(self, Xy, extras):
        (X, y), (key, *extras) = Xy, extras
        # make sure we aren't overwriting anything
        assert len(X) == len(y) == 0, f"X.keys={X.keys()}, y.keys={y.keys()}"
        key, consume = jax.random.split(key)
        Xy = self.finalize_fn(consume, *extras)
        return Xy, tuple([key] + extras)


def _rename_links(d: dict[str, dict], names: list[str]) -> dict[int, dict]:
    for key in list(d.keys()):
        if key in names:
            d[str(names.index(key))] = d.pop(key)
        else:
            warnings.warn(
                f"The key `{key}` was not found in names `{names}`. "
                "It will not be renamed."
            )

    return d


class Names2Indices:
    def __init__(self, sys_noimu: base.System) -> None:
        self.sys_noimu = sys_noimu

    def __call__(self, Xy, extras):
        (X, y), extras = Xy, extras
        X = _rename_links(X, self.sys_noimu.link_names)
        y = _rename_links(y, self.sys_noimu.link_names)
        return (X, y), extras


class JointAxisSensor:
    def __init__(self, sys: base.System, **kwargs):
        self.sys = sys
        self.kwargs = kwargs

    def __call__(self, Xy, extras):
        (X, y), (key, q, x, sys_x) = Xy, extras
        key, consume = jax.random.split(key)
        X_joint_axes = sensors.joint_axes(
            self.sys, x, sys_x, key=consume, **self.kwargs
        )
        X = utils.dict_union(X, X_joint_axes)
        return (X, y), (key, q, x, sys_x)


class RelPose:
    def __init__(self, sys: base.System):
        self.sys = sys

    def __call__(self, Xy, extras):
        (X, y), (key, q, x, sys_x) = Xy, extras
        y_relpose = sensors.rel_pose(self.sys, x, sys_x)
        y = utils.dict_union(y, y_relpose)
        return (X, y), (key, q, x, sys_x)


class RootIncl:
    def __init__(self, sys: base.System, **kwargs):
        self.sys = sys
        self.kwargs = kwargs

    def __call__(self, Xy, extras):
        (X, y), (key, q, x, sys_x) = Xy, extras
        y_root_incl = sensors.root_incl(self.sys, x, sys_x, **self.kwargs)
        y = utils.dict_union(y, y_root_incl)
        return (X, y), (key, q, x, sys_x)


class RootFull:
    def __init__(self, sys: base.System, **kwargs):
        self.sys = sys
        self.kwargs = kwargs

    def __call__(self, Xy, extras):
        (X, y), (key, q, x, sys_x) = Xy, extras
        y_root_incl = sensors.root_full(self.sys, x, sys_x, **self.kwargs)
        y = utils.dict_union(y, y_root_incl)
        return (X, y), (key, q, x, sys_x)


_default_imu_kwargs = dict(
    noisy=True,
    low_pass_filter_pos_f_cutoff=13.5,
    low_pass_filter_rot_cutoff=16.0,
)


class IMU:
    def __init__(self, **imu_kwargs):
        self.kwargs = _default_imu_kwargs.copy()
        self.kwargs.update(imu_kwargs)

    def __call__(self, Xy: types.Xy, extras: types.OutputExtras):
        (X, y), (key, q, x, sys) = Xy, extras
        key, consume = jax.random.split(key)
        X_imu = _imu_data(consume, x, sys, **self.kwargs)
        X = utils.dict_union(X, X_imu)
        return (X, y), (key, q, x, sys)


def _imu_data(key, xs, sys_xs, **kwargs) -> dict:
    sys_noimu, imu_attachment = sys_xs.make_sys_noimu()
    inv_imu_attachment = {val: key for key, val in imu_attachment.items()}
    X = {}
    N = xs.shape()
    for segment in sys_noimu.link_names:
        if segment in inv_imu_attachment:
            imu = inv_imu_attachment[segment]
            key, consume = jax.random.split(key)
            imu_measurements = sensors.imu(
                xs=xs.take(sys_xs.name_to_idx(imu), 1),
                gravity=sys_xs.gravity,
                dt=sys_xs.dt,
                key=consume,
                **kwargs,
            )
        else:
            imu_measurements = {
                "acc": jnp.zeros(
                    (
                        N,
                        3,
                    )
                ),
                "gyr": jnp.zeros(
                    (
                        N,
                        3,
                    )
                ),
            }
        X[segment] = imu_measurements
    return X


P_rot, P_pos = 100.0, 250.0
_P_gains = {
    "free": jnp.array(3 * [P_rot] + 3 * [P_pos]),
    "free_2d": jnp.array(1 * [P_rot] + 2 * [P_pos]),
    "px": jnp.array([P_pos]),
    "py": jnp.array([P_pos]),
    "pz": jnp.array([P_pos]),
    "rx": jnp.array([P_rot]),
    "ry": jnp.array([P_rot]),
    "rz": jnp.array([P_rot]),
    "rr": jnp.array([P_rot]),
    # primary, residual
    "rr_imp": jnp.array([P_rot, P_rot]),
    "cor": jnp.array(3 * [P_rot] + 6 * [P_pos]),
    "spherical": jnp.array(3 * [P_rot]),
    "p3d": jnp.array(3 * [P_pos]),
    "saddle": jnp.array([P_rot, P_rot]),
    "rsaddle": jnp.array([P_rot, P_rot]),
    "frozen": jnp.array([]),
    "suntay": jnp.array([P_rot]),
}


class DynamicalSimulation:
    def __init__(
        self,
        custom_P_gains: dict[str, jax.Array] = dict(),
        unactuated_subsystems: list[str] = [],
        return_q_ref: bool = False,
        overwrite_q_ref: Optional[tuple[jax.Array, dict[str, slice]]] = None,
        **unroll_kwargs,
    ):
        self.unactuated_links = unactuated_subsystems
        self.custom_P_gains = custom_P_gains
        self.return_q_ref = return_q_ref
        self.overwrite_q_ref = overwrite_q_ref
        self.unroll_kwargs = unroll_kwargs

    @staticmethod
    def assert_test_system(sys: base.System) -> None:
        "test that system has no zero mass leaf bodies and no joints without damping"

        def f(_, __, n, m, d):
            is_leaf_body = len(sys.children(n)) == 0
            if is_leaf_body:
                assert d.size == 0 or m > 0, (
                    "Dynamic simulation is set to `True` which requires masses >= 0, "
                    f"but found body `{n}` with mass={float(m[0])}. This can lead to "
                    "NaNs."
                )

            assert d.size == 0 or all(d > 0.0), (
                "Dynamic simulation is set to `True` which requires dampings > 0, "
                f"but found body `{n}` with damping={d}. This can lead to NaNs."
            )

        sys.scan(f, "lld", sys.link_names, sys.links.inertia.mass, sys.link_damping)

    def __call__(
        self, Xy: types.Xy, extras: types.OutputExtras
    ) -> tuple[types.Xy, types.OutputExtras]:
        (X, y), (key, q, _, sys_x) = Xy, extras
        idx_map_q = sys_x.idx_map("q")

        if self.overwrite_q_ref is not None:
            q, idx_map_q = self.overwrite_q_ref
            assert q.shape[-1] == sum([s.stop - s.start for s in idx_map_q.values()])

        sys_q_ref = sys_x
        if len(self.unactuated_links) > 0:
            sys_q_ref = sys_x.delete_system(self.unactuated_links)

        q_ref = []
        p_gains_list = []
        q = q.T

        def build_q_ref(_, __, name, link_type):
            q_ref.append(q[idx_map_q[name]])

            if link_type in self.custom_P_gains:
                p_gain_this_link = self.custom_P_gains[link_type]
            elif link_type in _P_gains:
                p_gain_this_link = _P_gains[link_type]
            else:
                raise RuntimeError(
                    f"Please proved gain parameters for the joint typ `{link_type}`"
                    " via the argument `custom_P_gains: dict[str, Array]`"
                )

            required_qd_size = base.QD_WIDTHS[link_type]
            assert (
                required_qd_size == p_gain_this_link.size
            ), f"The gain parameters must be of qd_size=`{required_qd_size}`"
            f" but got `{p_gain_this_link.size}`. This happened for the link "
            f"`{name}` of type `{link_type}`."
            p_gains_list.append(p_gain_this_link)

        sys_q_ref.scan(build_q_ref, "ll", sys_q_ref.link_names, sys_q_ref.link_types)
        q_ref, p_gains_array = jnp.concatenate(q_ref).T, jnp.concatenate(p_gains_list)

        # perform dynamical simulation
        states = pd_control._unroll_dynamics_pd_control(
            sys_x, q_ref, p_gains_array, sys_q_ref=sys_q_ref, **self.unroll_kwargs
        )

        if self.return_q_ref:
            X = utils.dict_union(X, dict(q_ref=q_ref))

        return (X, y), (key, states.q, states.x, sys_x)


def _flatten(seq: list):
    seq = tree_utils.tree_batch(seq, backend=None)
    seq = tree_utils.batch_concat_acme(seq, num_batch_dims=3).transpose((1, 2, 0, 3))
    return seq


def _expand_dt(X: dict, T: int):
    dt = X.pop("dt", None)
    if dt is not None:
        if isinstance(dt, np.ndarray):
            numpy = np
        else:
            numpy = jnp
        dt = numpy.repeat(dt[:, None, :], T, axis=1)
        for seg in X:
            X[seg]["dt"] = dt
    return X


def _expand_then_flatten(Xy):
    X, y = Xy
    gyr = X["0"]["gyr"]

    batched = True
    if gyr.ndim == 2:
        batched = False
        X, y = tree_utils.add_batch_dim((X, y))

    X = _expand_dt(X, gyr.shape[-2])

    N = len(X)

    def dict_to_tuple(d: dict[str, jax.Array]):
        tup = (d["acc"], d["gyr"])
        if "joint_axes" in d:
            tup = tup + (d["joint_axes"],)
        if "dt" in d:
            tup = tup + (d["dt"],)
        return tup

    X = [dict_to_tuple(X[str(i)]) for i in range(N)]
    y = [y[str(i)] for i in range(N)]

    X, y = _flatten(X), _flatten(y)
    if not batched:
        X, y = jax.tree_map(lambda arr: arr[0], (X, y))
    return X, y


class GeneratorTrafoLambda:
    def __init__(self, f, input: bool = False):
        self.f = f
        self.input = input

    def __call__(self, gen):
        if self.input:

            def _gen(*args):
                return gen(*self.f(*args))

        else:

            def _gen(*args):
                return self.f(gen(*args))

        return _gen


def GeneratorTrafoExpandFlatten(gen, jit: bool = False):
    if jit:
        return GeneratorTrafoLambda(jax.jit(_expand_then_flatten))(gen)
    return GeneratorTrafoLambda(_expand_then_flatten)(gen)
