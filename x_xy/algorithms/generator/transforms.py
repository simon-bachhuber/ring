import jax
import jax.numpy as jnp

from ... import base
from ... import maths
from ..sensors import imu as imu_fn
from ..sensors import joint_axes
from ..sensors import rel_pose
from .types import FINALIZE_FN
from .types import GeneratorTrafo
from .types import GeneratorWithInputExtras
from .types import GeneratorWithInputOutputExtras
from .types import GeneratorWithOutputExtras
from .types import SETUP_FN


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
            _, (key, *extras) = gen(*args)
            key, consume = jax.random.split(key)
            Xy = self.finalize_fn(consume, *extras)
            return Xy, tuple(list(key) + extras)

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


class GeneratorTrafoRandomizeTransform1Rot:
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


def _update_X(X: dict[str, dict], X_mixin: dict[str, dict]) -> dict:
    for segment in X_mixin:
        if segment not in X:
            X[segment] = X_mixin[segment]
        else:
            assert isinstance(X[segment], dict)

            for key in X_mixin[segment]:
                assert key not in X[segment]

            X[segment].update(X_mixin[segment])
    return X


class GeneratorTrafoJointAxisSensor(GeneratorTrafo):
    def __init__(self, sys: base.System):
        self.sys = sys

    def __call__(self, gen):
        def _gen(*args):
            (X, y), (key, q, x, sys_x) = gen(*args)
            X_joint_axes = joint_axes(self.sys, x, sys_x)
            X = _update_X(X, X_joint_axes)
            return (X, y), (key, q, x, sys_x)

        return _gen


class GeneratorTrafoRelPose(GeneratorTrafo):
    def __init__(self, sys: base.System):
        self.sys = sys

    def __call__(self, gen):
        def _gen(*args):
            (X, y), (key, q, x, sys_x) = gen(*args)
            y_relpose = rel_pose(self.sys, x, sys_x)
            y = _update_X(y, y_relpose)
            return (X, y), (key, q, x, sys_x)

        return _gen


class GeneratorTrafoIMU(GeneratorTrafo):
    def __call__(self, gen: GeneratorWithOutputExtras | GeneratorWithInputOutputExtras):
        def _gen(*args):
            (X, y), (key, q, x, sys) = gen(*args)
            key, consume = jax.random.split(key)
            X_imu = _imu_data(consume, x, sys)
            X = _update_X(X, X_imu)
            return (X, y), (key, q, x, sys)

        return _gen


def _imu_data(
    key,
    xs,
    sys_xs,
) -> dict:
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
                low_pass_filter_pos_f_cutoff=13.5,
                low_pass_filter_rot_alpha=0.5,
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
