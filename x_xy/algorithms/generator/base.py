from typing import Callable, Optional, Sequence
import warnings

import jax
import jax.numpy as jnp
import tree_utils

from . import motion_artifacts
from ... import base
from ...scan import scan_sys
from ...utils import hdf5_save
from ...utils import to_list
from ..jcalc import _init_joint_params
from ..jcalc import _joint_types
from ..jcalc import RCMG_Config
from ..kinematics import forward_kinematics_transforms
from .batch import batch_generators_eager
from .batch import batch_generators_eager_to_list
from .batch import batch_generators_lazy
from .transforms import GeneratorTrafoDropout
from .transforms import GeneratorTrafoDynamicalSimulation
from .transforms import GeneratorTrafoFinalizeFn
from .transforms import GeneratorTrafoIMU
from .transforms import GeneratorTrafoJointAxisSensor
from .transforms import GeneratorTrafoLambda
from .transforms import GeneratorTrafoRandomizePositions
from .transforms import GeneratorTrafoRelPose
from .transforms import GeneratorTrafoRootIncl
from .transforms import GeneratorTrafoSetupFn
from .types import FINALIZE_FN
from .types import Generator
from .types import GeneratorTrafo
from .types import GeneratorWithInputExtras
from .types import GeneratorWithInputOutputExtras
from .types import GeneratorWithOutputExtras
from .types import OutputExtras
from .types import PRNGKey
from .types import SETUP_FN
from .types import Xy


def _copy_kwargs(kwargs: dict | None) -> dict:
    return dict() if kwargs is None else kwargs.copy()


def build_generator(
    sys: base.System | list[base.System],
    config: RCMG_Config | list[RCMG_Config] = RCMG_Config(),
    setup_fn: Optional[SETUP_FN] = None,
    finalize_fn: Optional[FINALIZE_FN] = None,
    add_X_imus: bool = False,
    add_X_imus_kwargs: Optional[dict] = None,
    add_X_jointaxes: bool = False,
    add_X_dropout: Optional[dict[str, tuple[float, float]]] = None,
    add_y_relpose: bool = False,
    add_y_rootincl: bool = False,
    sys_ml: Optional[base.System] = None,
    randomize_positions: bool = False,
    randomize_motion_artifacts: bool = False,
    randomize_joint_params: bool = False,
    imu_motion_artifacts: bool = False,
    imu_motion_artifacts_kwargs: Optional[dict] = None,
    dynamic_simulation: bool = False,
    dynamic_simulation_kwargs: Optional[dict] = None,
    output_transform: Optional[Callable] = None,
    keep_output_extras: bool = False,
    eager: bool = False,
    aslist: bool = False,
    ashdf5: Optional[str] = None,
    seed: Optional[int] = None,
    sizes: Optional[int | list[int]] = None,
    batchsize: Optional[int] = None,
    _compat: bool = False,
) -> Generator | GeneratorWithOutputExtras | None | list:
    """
    If `eager` then returns numpy, else jax.
    """
    # capture all function args
    kwargs = locals()

    batch = False
    if (
        sizes is not None
        or isinstance(sys, list)
        or isinstance(config, list)
        or eager
        or aslist
        or ashdf5 is not None
    ):
        batch = True

    if batch:
        assert sizes is not None

        if ashdf5 is not None:
            assert not aslist
            aslist = True

        if aslist:
            assert eager
        if eager and not aslist:
            assert batchsize is not None
        else:
            assert (
                batchsize is None
            ), "Use `sizes` instead to provide the sizes per batch."
        if eager:
            assert seed is not None

        sys, config = to_list(sys), to_list(config)

        if kwargs["sys_ml"] is None and len(sys) > 1:
            warnings.warn(
                "Batched simulation with multiple systems but no explicit `sys_ml`"
            )
            kwargs["sys_ml"] = sys[0]

        gens = []
        kwargs["eager"] = False
        kwargs["aslist"] = False
        kwargs["ashdf5"] = None
        kwargs["sizes"] = None
        for _sys in sys:
            for _config in config:
                kwargs["sys"] = _sys
                kwargs["config"] = _config
                gens.append(build_generator(**kwargs))

        if eager:
            if aslist:
                # returns pytree of numpy arrays
                data = batch_generators_eager_to_list(gens, sizes, seed=seed)
                if ashdf5 is None:
                    return data
                else:
                    data = tree_utils.tree_batch(data)
                    hdf5_save(ashdf5, data, overwrite=True)
            else:
                return batch_generators_eager(gens, sizes, batchsize, seed=seed)
        else:
            return batch_generators_lazy(gens, sizes)

        # if `batch` is True, then this function must always recursively call
        # itself, so we exit here; all work is done
        return

    # end of batch generator logic - non-batched build_generator logic starts
    assert config.is_feasible()

    # re-enable old finalize_fn logic such that all tests can still work
    if _compat:

        def finalize_fn(key, q, x, sys):
            return q, x

    imu_motion_artifacts_kwargs = _copy_kwargs(imu_motion_artifacts_kwargs)
    dynamic_simulation_kwargs = _copy_kwargs(dynamic_simulation_kwargs)
    add_X_imus_kwargs = _copy_kwargs(add_X_imus_kwargs)

    # default kwargs values
    if "hide_injected_bodies" not in imu_motion_artifacts_kwargs:
        imu_motion_artifacts_kwargs["hide_injected_bodies"] = True

    if sys_ml is None:
        sys_ml = sys

    if add_X_jointaxes or add_y_relpose or add_y_rootincl:
        if len(sys_ml.findall_imus()) > 0:
            warnings.warn("Automatically removed the IMUs from `sys_ml`.")

            from x_xy.subpkgs import sys_composer

            sys_noimu, _ = sys_composer.make_sys_noimu(sys_ml)
        else:
            sys_noimu = sys_ml

    unactuated_subsystems = []
    if imu_motion_artifacts:
        assert dynamic_simulation
        unactuated_subsystems = motion_artifacts.unactuated_subsystem(sys)
        sys = motion_artifacts.inject_subsystems(sys, **imu_motion_artifacts_kwargs)
        assert "unactuated_subsystems" not in dynamic_simulation_kwargs
        dynamic_simulation_kwargs["unactuated_subsystems"] = unactuated_subsystems

        if not randomize_motion_artifacts:
            warnings.warn(
                "`imu_motion_artifacts` is enabled but not `randomize_motion_artifacts`"
            )

        if "hide_injected_bodies" in imu_motion_artifacts_kwargs:
            if imu_motion_artifacts_kwargs["hide_injected_bodies"]:
                warnings.warn(
                    "The flag `hide_injected_bodies` in `imu_motion_artifacts_kwargs` "
                    "is set. This will try to hide injected bodies. This feature is "
                    "experimental."
                )

        if "prob_rigid" in imu_motion_artifacts_kwargs:
            assert randomize_motion_artifacts, (
                "`prob_rigid` works by overwriting damping and stiffness parameters "
                "using the `randomize_motion_artifacts` flag, so it must be enabled."
            )

    noop = lambda gen: gen
    return GeneratorPipe(
        GeneratorTrafoSetupFn(_init_joint_params) if randomize_joint_params else noop,
        GeneratorTrafoRandomizePositions() if randomize_positions else noop,
        GeneratorTrafoSetupFn(
            motion_artifacts.setup_fn_randomize_damping_stiffness_factory(
                imu_motion_artifacts_kwargs.get("prob_rigid", 0.0)
            )
        )
        if (imu_motion_artifacts and randomize_motion_artifacts)
        else noop,
        GeneratorTrafoSetupFn(setup_fn) if setup_fn is not None else noop,
        GeneratorTrafoDynamicalSimulation(**dynamic_simulation_kwargs)
        if dynamic_simulation
        else noop,
        motion_artifacts.GeneratorTrafoHideInjectedBodies()
        if (
            imu_motion_artifacts and imu_motion_artifacts_kwargs["hide_injected_bodies"]
        )
        else noop,
        GeneratorTrafoFinalizeFn(finalize_fn) if finalize_fn is not None else noop,
        GeneratorTrafoIMU(**add_X_imus_kwargs) if add_X_imus else noop,
        GeneratorTrafoJointAxisSensor(sys_noimu) if add_X_jointaxes else noop,
        GeneratorTrafoDropout(add_X_dropout) if add_X_dropout is not None else noop,
        GeneratorTrafoRelPose(sys_noimu) if add_y_relpose else noop,
        GeneratorTrafoRootIncl(sys_noimu) if add_y_rootincl else noop,
        GeneratorTrafoRemoveInputExtras(sys),
        noop if keep_output_extras else GeneratorTrafoRemoveOutputExtras(),
        GeneratorTrafoLambda(output_transform, input=False)
        if output_transform is not None
        else noop,
    )(config)


def _generator_with_extras(
    config: RCMG_Config,
) -> GeneratorWithInputOutputExtras:
    def generator(key: PRNGKey, sys: base.System) -> tuple[Xy, OutputExtras]:
        if config.cor:
            sys = sys._replace_free_with_cor()

        key_start = key
        # build generalized coordintes vector `q`
        q_list = []

        def draw_q(key, __, link_type, link):
            joint_params = link.joint_params
            # limit scope
            joint_params = (
                joint_params[link_type]
                if link_type in joint_params
                else joint_params["default"]
            )
            if key is None:
                key = key_start
            key, key_t, key_value = jax.random.split(key, 3)
            draw_fn = _joint_types[link_type].rcmg_draw_fn
            if draw_fn is None:
                raise Exception(f"The joint type {link_type} has no draw fn specified.")
            q_link = draw_fn(config, key_t, key_value, sys.dt, joint_params)
            # even revolute and prismatic joints must be 2d arrays
            q_link = q_link if q_link.ndim == 2 else q_link[:, None]
            q_list.append(q_link)
            return key

        keys = scan_sys(sys, draw_q, "ll", sys.link_types, sys.links)
        # stack of keys; only the last key is unused
        key = keys[-1]

        q = jnp.concatenate(q_list, axis=1)

        # do forward kinematics
        x, _ = jax.vmap(forward_kinematics_transforms, (None, 0))(sys, q)

        Xy = ({}, {})
        return Xy, (key, q, x, sys)

    return generator


class GeneratorPipe:
    def __init__(self, *gen_trafos: Sequence[GeneratorTrafo]):
        self._gen_trafos = gen_trafos

    def __call__(
        self, config: RCMG_Config
    ) -> (
        GeneratorWithInputOutputExtras
        | GeneratorWithOutputExtras
        | GeneratorWithInputExtras
        | Generator
    ):
        gen = _generator_with_extras(config)
        for trafo in self._gen_trafos:
            gen = trafo(gen)
        return gen


class GeneratorTrafoRemoveInputExtras(GeneratorTrafo):
    def __init__(self, sys: base.System):
        self.sys = sys

    def __call__(
        self,
        gen: GeneratorWithInputExtras | GeneratorWithInputOutputExtras,
    ) -> Generator | GeneratorWithOutputExtras:
        def _gen(key):
            return gen(key, self.sys)

        return _gen


class GeneratorTrafoRemoveOutputExtras(GeneratorTrafo):
    def __call__(
        self,
        gen: GeneratorWithOutputExtras | GeneratorWithInputOutputExtras,
    ) -> Generator | GeneratorWithInputExtras:
        def _gen(*args):
            return gen(*args)[0]

        return _gen
