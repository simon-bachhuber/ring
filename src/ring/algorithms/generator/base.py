from functools import partial
from typing import Callable, Optional, Sequence
import warnings

import jax
import jax.numpy as jnp
import tqdm
import tree_utils

from ring import base
from ring import utils
from ring.algorithms import jcalc
from ring.algorithms import kinematics
from ring.algorithms.generator import batch
from ring.algorithms.generator import motion_artifacts
from ring.algorithms.generator import randomize
from ring.algorithms.generator import transforms
from ring.algorithms.generator import types


class RCMG:
    def __init__(
        self,
        sys: base.System | list[base.System],
        config: jcalc.MotionConfig | list[jcalc.MotionConfig] = jcalc.MotionConfig(),
        setup_fn: Optional[types.SETUP_FN] = None,
        finalize_fn: Optional[types.FINALIZE_FN] = None,
        add_X_imus: bool = False,
        add_X_imus_kwargs: Optional[dict] = None,
        add_X_jointaxes: bool = False,
        add_X_jointaxes_kwargs: Optional[dict] = None,
        add_y_relpose: bool = False,
        add_y_rootincl: bool = False,
        sys_ml: Optional[base.System] = None,
        randomize_positions: bool = False,
        randomize_motion_artifacts: bool = False,
        randomize_joint_params: bool = False,
        randomize_anchors: bool = False,
        randomize_anchors_kwargs: Optional[dict] = None,
        randomize_hz: bool = False,
        randomize_hz_kwargs: Optional[dict] = None,
        imu_motion_artifacts: bool = False,
        imu_motion_artifacts_kwargs: Optional[dict] = None,
        dynamic_simulation: bool = False,
        dynamic_simulation_kwargs: Optional[dict] = None,
        output_transform: Optional[Callable] = None,
        keep_output_extras: bool = False,
        use_link_number_in_Xy: bool = False,
    ) -> None:

        randomize_anchors_kwargs = _copy_kwargs(randomize_anchors_kwargs)
        randomize_hz_kwargs = _copy_kwargs(randomize_hz_kwargs)

        if randomize_hz:
            finalize_fn = randomize.randomize_hz_finalize_fn_factory(finalize_fn)

        partial_build_gen = partial(
            _build_generator_lazy,
            setup_fn=setup_fn,
            finalize_fn=finalize_fn,
            add_X_imus=add_X_imus,
            add_X_imus_kwargs=add_X_imus_kwargs,
            add_X_jointaxes=add_X_jointaxes,
            add_X_jointaxes_kwargs=add_X_jointaxes_kwargs,
            add_y_relpose=add_y_relpose,
            add_y_rootincl=add_y_rootincl,
            randomize_positions=randomize_positions,
            randomize_motion_artifacts=randomize_motion_artifacts,
            randomize_joint_params=randomize_joint_params,
            imu_motion_artifacts=imu_motion_artifacts,
            imu_motion_artifacts_kwargs=imu_motion_artifacts_kwargs,
            dynamic_simulation=dynamic_simulation,
            dynamic_simulation_kwargs=dynamic_simulation_kwargs,
            output_transform=output_transform,
            keep_output_extras=keep_output_extras,
            use_link_number_in_Xy=use_link_number_in_Xy,
        )

        sys, config = utils.to_list(sys), utils.to_list(config)

        if randomize_anchors:
            assert (
                len(sys) == 1
            ), "If `randomize_anchors`, then only one system is expected"
            sys = randomize.randomize_anchors(sys[0], **randomize_anchors_kwargs)

        if randomize_hz:
            sys, config = randomize.randomize_hz(sys, config, **randomize_hz_kwargs)
        else:
            # create zip
            N_sys = len(sys)
            sys = sum([len(config) * [s] for s in sys], start=[])
            config = N_sys * config
            assert len(sys) == len(config)

        if sys_ml is None:
            # TODO
            if False and len(sys) > 1:
                warnings.warn(
                    "Batched simulation with multiple systems but no explicit `sys_ml`"
                )
            sys_ml = sys[0]

        self.gens = []
        for _sys, _config in tqdm.tqdm(
            zip(sys, config), desc="building generators", total=len(sys)
        ):
            self.gens.append(partial_build_gen(sys=_sys, config=_config, sys_ml=sys_ml))

    def _to_data(self, sizes, seed):
        return batch.batch_generators_eager_to_list(self.gens, sizes, seed=seed)

    def to_list(self, sizes: int | list[int] = 1, seed: int = 1):
        return self._to_data(sizes, seed)

    def to_pickle(
        self,
        path: str,
        sizes: int | list[int] = 1,
        seed: int = 1,
        overwrite: bool = True,
    ) -> None:
        data = tree_utils.tree_batch(self._to_data(sizes, seed))
        utils.pickle_save(data, path, overwrite=overwrite)

    def to_hdf5(
        self,
        path: str,
        sizes: int | list[int] = 1,
        seed: int = 1,
        overwrite: bool = True,
    ) -> None:
        data = tree_utils.tree_batch(self._to_data(sizes, seed))
        utils.hdf5_save(path, data, overwrite=overwrite)

    def to_eager_gen(
        self,
        batchsize: int = 1,
        sizes: int | list[int] = 1,
        seed: int = 1,
        shuffle: bool = True,
    ) -> types.BatchedGenerator:
        return batch.batch_generators_eager(
            self.gens, sizes, batchsize, seed=seed, shuffle=shuffle
        )

    def to_lazy_gen(
        self, sizes: int | list[int] = 1, jit: bool = True
    ) -> types.BatchedGenerator:
        return batch.batch_generators_lazy(self.gens, sizes, jit=jit)

    @staticmethod
    def eager_gen_from_paths(
        paths: str | list[str],
        batchsize: int,
        include_samples: Optional[list[int]] = None,
        shuffle: bool = True,
        load_all_into_memory: bool = False,
        tree_transform=None,
    ) -> tuple[types.BatchedGenerator, int]:
        paths = utils.to_list(paths)
        return batch.batched_generator_from_paths(
            paths,
            batchsize,
            include_samples,
            shuffle,
            load_all_into_memory=load_all_into_memory,
            tree_transform=tree_transform,
        )


def _copy_kwargs(kwargs: dict | None) -> dict:
    return dict() if kwargs is None else kwargs.copy()


def _build_generator_lazy(
    sys: base.System,
    config: jcalc.MotionConfig,
    setup_fn: types.SETUP_FN | None,
    finalize_fn: types.FINALIZE_FN | None,
    add_X_imus: bool,
    add_X_imus_kwargs: dict | None,
    add_X_jointaxes: bool,
    add_X_jointaxes_kwargs: dict | None,
    add_y_relpose: bool,
    add_y_rootincl: bool,
    sys_ml: base.System,
    randomize_positions: bool,
    randomize_motion_artifacts: bool,
    randomize_joint_params: bool,
    imu_motion_artifacts: bool,
    imu_motion_artifacts_kwargs: dict | None,
    dynamic_simulation: bool,
    dynamic_simulation_kwargs: dict | None,
    output_transform: Callable | None,
    keep_output_extras: bool,
    use_link_number_in_Xy: bool,
) -> types.Generator | types.GeneratorWithOutputExtras:
    assert config.is_feasible()

    imu_motion_artifacts_kwargs = _copy_kwargs(imu_motion_artifacts_kwargs)
    dynamic_simulation_kwargs = _copy_kwargs(dynamic_simulation_kwargs)
    add_X_imus_kwargs = _copy_kwargs(add_X_imus_kwargs)
    add_X_jointaxes_kwargs = _copy_kwargs(add_X_jointaxes_kwargs)

    # default kwargs values
    if "hide_injected_bodies" not in imu_motion_artifacts_kwargs:
        imu_motion_artifacts_kwargs["hide_injected_bodies"] = True

    if add_X_jointaxes or add_y_relpose or add_y_rootincl:
        if len(sys_ml.findall_imus()) > 0:
            # warnings.warn("Automatically removed the IMUs from `sys_ml`.")
            sys_noimu, _ = sys_ml.make_sys_noimu()
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
            if imu_motion_artifacts_kwargs["hide_injected_bodies"] and False:
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
        transforms.GeneratorTrafoSetupFn(setup_fn) if setup_fn is not None else noop,
        (
            transforms.GeneratorTrafoSetupFn(jcalc._init_joint_params)
            if randomize_joint_params
            else noop
        ),
        transforms.GeneratorTrafoRandomizePositions() if randomize_positions else noop,
        (
            transforms.GeneratorTrafoSetupFn(
                motion_artifacts.setup_fn_randomize_damping_stiffness_factory(
                    prob_rigid=imu_motion_artifacts_kwargs.get("prob_rigid", 0.0),
                    all_imus_either_rigid_or_flex=imu_motion_artifacts_kwargs.get(
                        "all_imus_either_rigid_or_flex", False
                    ),
                    imus_surely_rigid=imu_motion_artifacts_kwargs.get(
                        "imus_surely_rigid", []
                    ),
                )
            )
            if (imu_motion_artifacts and randomize_motion_artifacts)
            else noop
        ),
        # all the generator trafors before this point execute in reverse order
        # to see this, consider gen[0] and gen[1]
        # the GeneratorPipe will unpack into the following:
        # gen[1] will unfold into
        # >>> sys = gen[1].setup_fn(sys)
        # >>> return gen[0](sys)
        # <-------------------- GENERATOR MIDDLE POINT ------------------------->
        # all the generator trafos after this point execute in order
        # >>> Xy, extras = gen[-2](*args)
        # >>> return gen[-1].finalize_fn(extras)
        (
            transforms.GeneratorTrafoDynamicalSimulation(**dynamic_simulation_kwargs)
            if dynamic_simulation
            else noop
        ),
        (
            motion_artifacts.GeneratorTrafoHideInjectedBodies()
            if (
                imu_motion_artifacts
                and imu_motion_artifacts_kwargs["hide_injected_bodies"]
            )
            else noop
        ),
        (
            transforms.GeneratorTrafoFinalizeFn(finalize_fn)
            if finalize_fn is not None
            else noop
        ),
        transforms.GeneratorTrafoIMU(**add_X_imus_kwargs) if add_X_imus else noop,
        (
            transforms.GeneratorTrafoJointAxisSensor(
                sys_noimu, **add_X_jointaxes_kwargs
            )
            if add_X_jointaxes
            else noop
        ),
        transforms.GeneratorTrafoRelPose(sys_noimu) if add_y_relpose else noop,
        transforms.GeneratorTrafoRootIncl(sys_noimu) if add_y_rootincl else noop,
        (
            transforms.GeneratorTrafoNames2Indices(sys_noimu)
            if use_link_number_in_Xy
            else noop
        ),
        GeneratorTrafoRemoveInputExtras(sys),
        noop if keep_output_extras else GeneratorTrafoRemoveOutputExtras(),
        (
            transforms.GeneratorTrafoLambda(output_transform, input=False)
            if output_transform is not None
            else noop
        ),
    )(config)


def _generator_with_extras(
    config: jcalc.MotionConfig,
) -> types.GeneratorWithInputOutputExtras:
    def generator(
        key: types.PRNGKey, sys: base.System
    ) -> tuple[types.Xy, types.OutputExtras]:
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
            draw_fn = jcalc.get_joint_model(link_type).rcmg_draw_fn
            if draw_fn is None:
                raise Exception(f"The joint type {link_type} has no draw fn specified.")
            q_link = draw_fn(config, key_t, key_value, sys.dt, joint_params)
            # even revolute and prismatic joints must be 2d arrays
            q_link = q_link if q_link.ndim == 2 else q_link[:, None]
            q_list.append(q_link)
            return key

        keys = sys.scan(draw_q, "ll", sys.link_types, sys.links)
        # stack of keys; only the last key is unused
        key = keys[-1]

        q = jnp.concatenate(q_list, axis=1)

        # do forward kinematics
        x, _ = jax.vmap(kinematics.forward_kinematics_transforms, (None, 0))(sys, q)

        Xy = ({}, {})
        return Xy, (key, q, x, sys)

    return generator


class GeneratorPipe:
    def __init__(self, *gen_trafos: Sequence[types.GeneratorTrafo]):
        self._gen_trafos = gen_trafos

    def __call__(
        self, config: jcalc.MotionConfig
    ) -> (
        types.GeneratorWithInputOutputExtras
        | types.GeneratorWithOutputExtras
        | types.GeneratorWithInputExtras
        | types.Generator
    ):
        gen = _generator_with_extras(config)
        for trafo in self._gen_trafos:
            gen = trafo(gen)
        return gen


class GeneratorTrafoRemoveInputExtras(types.GeneratorTrafo):
    def __init__(self, sys: base.System):
        self.sys = sys

    def __call__(
        self,
        gen: types.GeneratorWithInputExtras | types.GeneratorWithInputOutputExtras,
    ) -> types.Generator | types.GeneratorWithOutputExtras:
        def _gen(key):
            return gen(key, self.sys)

        return _gen


class GeneratorTrafoRemoveOutputExtras(types.GeneratorTrafo):
    def __call__(
        self,
        gen: types.GeneratorWithOutputExtras | types.GeneratorWithInputOutputExtras,
    ) -> types.Generator | types.GeneratorWithInputExtras:
        def _gen(*args):
            return gen(*args)[0]

        return _gen
