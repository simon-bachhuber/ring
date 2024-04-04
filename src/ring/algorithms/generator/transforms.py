from typing import Optional
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import tree_utils

from ring import base
from ring import maths
from ring import utils
from ring.algorithms import sensors
from ring.algorithms.generator import pd_control
from ring.algorithms.generator import types


class GeneratorTrafoLambda(types.GeneratorTrafo):
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


class GeneratorTrafoNames2Indices(types.GeneratorTrafo):
    def __init__(self, sys_noimu: base.System) -> None:
        self.sys_noimu = sys_noimu

    def __call__(self, gen: types.GeneratorWithInputOutputExtras):
        def _gen(*args):
            (X, y), extras = gen(*args)
            X = _rename_links(X, self.sys_noimu.link_names)
            y = _rename_links(y, self.sys_noimu.link_names)
            return (X, y), extras

        return _gen


class GeneratorTrafoSetupFn(types.GeneratorTrafo):
    def __init__(self, setup_fn: types.SETUP_FN):
        self.setup_fn = setup_fn

    def __call__(
        self,
        gen: types.GeneratorWithInputExtras | types.GeneratorWithInputOutputExtras,
    ) -> types.GeneratorWithInputExtras | types.GeneratorWithInputOutputExtras:
        def _gen(key, sys):
            key, consume = jax.random.split(key)
            sys = self.setup_fn(consume, sys)
            return gen(key, sys)

        return _gen


class GeneratorTrafoFinalizeFn(types.GeneratorTrafo):
    def __init__(self, finalize_fn: types.FINALIZE_FN):
        self.finalize_fn = finalize_fn

    def __call__(
        self,
        gen: types.GeneratorWithOutputExtras | types.GeneratorWithInputOutputExtras,
    ) -> types.GeneratorWithOutputExtras | types.GeneratorWithInputOutputExtras:
        def _gen(*args):
            (X, y), (key, *extras) = gen(*args)
            # make sure we aren't overwriting anything
            assert len(X) == len(y) == 0, f"X.keys={X.keys()}, y.keys={y.keys()}"
            key, consume = jax.random.split(key)
            Xy = self.finalize_fn(consume, *extras)
            return Xy, tuple([key] + extras)

        return _gen


class GeneratorTrafoRandomizePositions(types.GeneratorTrafo):
    def __call__(
        self,
        gen: types.GeneratorWithInputExtras | types.GeneratorWithInputOutputExtras,
    ) -> types.GeneratorWithInputExtras | types.GeneratorWithInputOutputExtras:
        return GeneratorTrafoSetupFn(_setup_fn_randomize_positions)(gen)


def _setup_fn_randomize_positions(key: jax.Array, sys: base.System) -> base.System:
    ts = sys.links.transform1

    for i in range(sys.num_links()):
        link = sys.links[i]
        key, new_pos = _draw_pos_uniform(key, link.pos_min, link.pos_max)
        ts = ts.index_set(i, ts[i].replace(pos=new_pos))

    return sys.replace(links=sys.links.replace(transform1=ts))


def _draw_pos_uniform(key, pos_min, pos_max):
    key, c1, c2, c3 = jax.random.split(key, num=4)
    pos = jnp.array(
        [
            jax.random.uniform(c1, minval=pos_min[0], maxval=pos_max[0]),
            jax.random.uniform(c2, minval=pos_min[1], maxval=pos_max[1]),
            jax.random.uniform(c3, minval=pos_min[2], maxval=pos_max[2]),
        ]
    )
    return key, pos


class GeneratorTrafoRandomizeTransform1Rot(types.GeneratorTrafo):
    def __init__(self, maxval_deg: float):
        self.maxval = jnp.deg2rad(maxval_deg)

    def __call__(self, gen):
        setup_fn = lambda key, sys: _setup_fn_randomize_transform1_rot(
            key, sys, self.maxval
        )
        return GeneratorTrafoSetupFn(setup_fn)(gen)


def _setup_fn_randomize_transform1_rot(
    key, sys, maxval: float, not_imus: bool = True
) -> base.System:
    new_transform1 = sys.links.transform1.replace(
        rot=maths.quat_random(key, (sys.num_links(),), maxval=maxval)
    )
    if not_imus:
        imus = [name for name in sys.link_names if name[:3] == "imu"]
        new_rot = new_transform1.rot
        for imu in imus:
            new_rot = new_rot.at[sys.name_to_idx(imu)].set(jnp.array([1.0, 0, 0, 0]))
        new_transform1 = new_transform1.replace(rot=new_rot)
    return sys.replace(links=sys.links.replace(transform1=new_transform1))


class GeneratorTrafoJointAxisSensor(types.GeneratorTrafo):
    def __init__(self, sys: base.System, **kwargs):
        self.sys = sys
        self.kwargs = kwargs

    def __call__(self, gen):
        def _gen(*args):
            (X, y), (key, q, x, sys_x) = gen(*args)
            key, consume = jax.random.split(key)
            X_joint_axes = sensors.joint_axes(
                self.sys, x, sys_x, key=consume, **self.kwargs
            )
            X = utils.dict_union(X, X_joint_axes)
            return (X, y), (key, q, x, sys_x)

        return _gen


class GeneratorTrafoRelPose(types.GeneratorTrafo):
    def __init__(self, sys: base.System):
        self.sys = sys

    def __call__(self, gen):
        def _gen(*args):
            (X, y), (key, q, x, sys_x) = gen(*args)
            y_relpose = sensors.rel_pose(self.sys, x, sys_x)
            y = utils.dict_union(y, y_relpose)
            return (X, y), (key, q, x, sys_x)

        return _gen


class GeneratorTrafoRootIncl(types.GeneratorTrafo):
    def __init__(self, sys: base.System):
        self.sys = sys

    def __call__(self, gen):
        def _gen(*args):
            (X, y), (key, q, x, sys_x) = gen(*args)
            y_root_incl = sensors.root_incl(self.sys, x, sys_x)
            y = utils.dict_union(y, y_root_incl)
            return (X, y), (key, q, x, sys_x)

        return _gen


_default_imu_kwargs = dict(
    noisy=True,
    low_pass_filter_pos_f_cutoff=13.5,
    low_pass_filter_rot_cutoff=16.0,
)


class GeneratorTrafoIMU(types.GeneratorTrafo):
    def __init__(self, **imu_kwargs):
        self.kwargs = _default_imu_kwargs.copy()
        self.kwargs.update(imu_kwargs)

    def __call__(
        self,
        gen: types.GeneratorWithOutputExtras | types.GeneratorWithInputOutputExtras,
    ):
        def _gen(*args):
            (X, y), (key, q, x, sys) = gen(*args)
            key, consume = jax.random.split(key)
            X_imu = _imu_data(consume, x, sys, **self.kwargs)
            X = utils.dict_union(X, X_imu)
            return (X, y), (key, q, x, sys)

        return _gen


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
    "frozen": jnp.array([]),
    "suntay": jnp.array([P_rot]),
}


class GeneratorTrafoDynamicalSimulation(types.GeneratorTrafo):
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

    def __call__(self, gen):
        def _gen(*args):
            (X, y), (key, q, _, sys_x) = gen(*args)
            idx_map_q = sys_x.idx_map("q")

            if self.overwrite_q_ref is not None:
                q, idx_map_q = self.overwrite_q_ref
                assert q.shape[-1] == sum(
                    [s.stop - s.start for s in idx_map_q.values()]
                )

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

            sys_q_ref.scan(
                build_q_ref, "ll", sys_q_ref.link_names, sys_q_ref.link_types
            )
            q_ref, p_gains_array = jnp.concatenate(q_ref).T, jnp.concatenate(
                p_gains_list
            )

            # perform dynamical simulation
            states = pd_control._unroll_dynamics_pd_control(
                sys_x, q_ref, p_gains_array, sys_q_ref=sys_q_ref, **self.unroll_kwargs
            )

            if self.return_q_ref:
                X = utils.dict_union(X, dict(q_ref=q_ref))

            return (X, y), (key, states.q, states.x, sys_x)

        return _gen


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


def _expand_then_flatten(args):
    X, y = args
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


def GeneratorTrafoExpandFlatten(gen, jit: bool = False):
    if jit:
        return GeneratorTrafoLambda(jax.jit(_expand_then_flatten))(gen)
    return GeneratorTrafoLambda(_expand_then_flatten)(gen)
