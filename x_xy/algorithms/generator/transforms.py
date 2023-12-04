from typing import Optional

import jax
import jax.numpy as jnp

from ... import base
from ... import maths
from ...scan import scan_sys
from ...utils import dict_union
from ..sensors import imu as imu_fn
from ..sensors import joint_axes
from ..sensors import rel_pose
from ..sensors import root_incl
from .pd_control import _unroll_dynamics_pd_control
from .types import FINALIZE_FN
from .types import GeneratorTrafo
from .types import GeneratorWithInputExtras
from .types import GeneratorWithInputOutputExtras
from .types import GeneratorWithOutputExtras
from .types import SETUP_FN


def _dropout_imu_jointaxes_factory(dropout_rates: dict[str, tuple[float, float]]):
    """
    Args:
        dropout_rates: {'seg': (imu_rate, joint_axes_rate)}

    Returns:
        Function: (key, X) -> X

    """

    def _X_transform(key, X):
        for segments, (imu_rate, jointaxes_rate) in dropout_rates.items():
            key, c1, c2 = jax.random.split(key, 3)
            factor_imu = jax.random.bernoulli(c1, p=(1 - imu_rate)).astype(int)
            factor_jointaxes = jax.random.bernoulli(c2, p=(1 - jointaxes_rate)).astype(
                int
            )

            for gyraccmag in ["gyr", "acc", "mag"]:
                if gyraccmag in X[segments]:
                    X[segments][gyraccmag] *= factor_imu

            if "joint_axes" in X[segments]:
                X[segments]["joint_axes"] *= factor_jointaxes
        return X

    return _X_transform


class GeneratorTrafoDropout(GeneratorTrafo):
    def __init__(self, dropout_rates: dict[str, tuple[float, float]]):
        "dropout_rates: {'seg': (imu_rate, joint_axes_rate)}"
        self.dropout_rates = dropout_rates

    def __call__(self, gen):
        _X_transform = _dropout_imu_jointaxes_factory(self.dropout_rates)

        def _gen(*args):
            # X: dict[str, dict[str, Array]]
            (X, y), (key, *extras) = gen(*args)
            key, consume = jax.random.split(key)
            X = _X_transform(consume, X)
            return (X, y), tuple([key] + extras)

        return _gen


class GeneratorTrafoLambda(GeneratorTrafo):
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


class GeneratorTrafoSetupFn(GeneratorTrafo):
    def __init__(self, setup_fn: SETUP_FN):
        self.setup_fn = setup_fn

    def __call__(
        self,
        gen: GeneratorWithInputExtras | GeneratorWithInputOutputExtras,
    ) -> GeneratorWithInputExtras | GeneratorWithInputOutputExtras:
        def _gen(key, sys):
            key, consume = jax.random.split(key)
            sys = self.setup_fn(consume, sys)
            return gen(key, sys)

        return _gen


class GeneratorTrafoFinalizeFn(GeneratorTrafo):
    def __init__(self, finalize_fn: FINALIZE_FN):
        self.finalize_fn = finalize_fn

    def __call__(
        self,
        gen: GeneratorWithOutputExtras | GeneratorWithInputOutputExtras,
    ) -> GeneratorWithOutputExtras | GeneratorWithInputOutputExtras:
        def _gen(*args):
            (X, y), (key, *extras) = gen(*args)
            # make sure we aren't overwriting anything
            assert len(X) == len(y) == 0, f"X.keys={X.keys()}, y.keys={y.keys()}"
            key, consume = jax.random.split(key)
            Xy = self.finalize_fn(consume, *extras)
            return Xy, tuple([key] + extras)

        return _gen


class GeneratorTrafoRandomizePositions(GeneratorTrafo):
    def __call__(
        self,
        gen: GeneratorWithInputExtras | GeneratorWithInputOutputExtras,
    ) -> GeneratorWithInputExtras | GeneratorWithInputOutputExtras:
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


class GeneratorTrafoRandomizeTransform1Rot(GeneratorTrafo):
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


class GeneratorTrafoJointAxisSensor(GeneratorTrafo):
    def __init__(self, sys: base.System):
        self.sys = sys

    def __call__(self, gen):
        def _gen(*args):
            (X, y), (key, q, x, sys_x) = gen(*args)
            X_joint_axes = joint_axes(self.sys, x, sys_x)
            X = dict_union(X, X_joint_axes)
            return (X, y), (key, q, x, sys_x)

        return _gen


class GeneratorTrafoRelPose(GeneratorTrafo):
    def __init__(self, sys: base.System):
        self.sys = sys

    def __call__(self, gen):
        def _gen(*args):
            (X, y), (key, q, x, sys_x) = gen(*args)
            y_relpose = rel_pose(self.sys, x, sys_x)
            y = dict_union(y, y_relpose)
            return (X, y), (key, q, x, sys_x)

        return _gen


class GeneratorTrafoRootIncl(GeneratorTrafo):
    def __init__(self, sys: base.System):
        self.sys = sys

    def __call__(self, gen):
        def _gen(*args):
            (X, y), (key, q, x, sys_x) = gen(*args)
            y_root_incl = root_incl(self.sys, x, sys_x)
            y = dict_union(y, y_root_incl)
            return (X, y), (key, q, x, sys_x)

        return _gen


class GeneratorTrafoIMU(GeneratorTrafo):
    def __init__(self, has_magnetometer: bool = False):
        self.has_magnetometer = has_magnetometer

    def __call__(self, gen: GeneratorWithOutputExtras | GeneratorWithInputOutputExtras):
        def _gen(*args):
            (X, y), (key, q, x, sys) = gen(*args)
            key, consume = jax.random.split(key)
            X_imu = _imu_data(consume, x, sys, self.has_magnetometer)
            X = dict_union(X, X_imu)
            return (X, y), (key, q, x, sys)

        return _gen


def _imu_data(key, xs, sys_xs, has_magnetometer) -> dict:
    # TODO
    from x_xy.subpkgs import sys_composer

    sys_noimu, imu_attachment = sys_composer.make_sys_noimu(sys_xs)
    inv_imu_attachment = {val: key for key, val in imu_attachment.items()}
    X = {}
    N = xs.shape()
    for segment in sys_noimu.link_names:
        if segment in inv_imu_attachment:
            imu = inv_imu_attachment[segment]
            key, consume = jax.random.split(key)
            imu_measurements = imu_fn(
                xs.take(sys_xs.name_to_idx(imu), 1),
                sys_xs.gravity,
                sys_xs.dt,
                consume,
                noisy=False,
                low_pass_filter_pos_f_cutoff=13.5,
                low_pass_filter_rot_cutoff=20.0,
                has_magnetometer=has_magnetometer,
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


P_rot, P_pos = 50.0, 200.0
_P_gains = {
    "free": jnp.array(3 * [P_rot] + 3 * [P_pos]),
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
}


class GeneratorTrafoDynamicalSimulation(GeneratorTrafo):
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
        from x_xy.subpkgs import sys_composer

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
                sys_q_ref = sys_composer.delete_subsystem(sys_x, self.unactuated_links)

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

            scan_sys(
                sys_q_ref, build_q_ref, "ll", sys_q_ref.link_names, sys_q_ref.link_types
            )
            q_ref, p_gains_array = jnp.concatenate(q_ref).T, jnp.concatenate(
                p_gains_list
            )

            # perform dynamical simulation
            states = _unroll_dynamics_pd_control(
                sys_x, q_ref, p_gains_array, sys_q_ref=sys_q_ref, **self.unroll_kwargs
            )

            if self.return_q_ref:
                X = dict_union(X, dict(q_ref=q_ref))

            return (X, y), (key, states.q, states.x, sys_x)

        return _gen
