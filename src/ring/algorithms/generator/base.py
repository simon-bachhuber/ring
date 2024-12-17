from dataclasses import replace
from functools import partial
import random
from typing import Callable, Optional
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import tree_utils
from tree_utils import PyTree

from ring import base
from ring import utils
from ring.algorithms import jcalc
from ring.algorithms import kinematics
from ring.algorithms.generator import batch
from ring.algorithms.generator import finalize_fns
from ring.algorithms.generator import motion_artifacts
from ring.algorithms.generator import setup_fns
from ring.algorithms.generator import types


class RCMG:
    def __init__(
        self,
        sys: base.System | list[base.System],
        config: jcalc.MotionConfig | list[jcalc.MotionConfig] = jcalc.MotionConfig(),
        setup_fn: Optional[types.SETUP_FN] = None,
        finalize_fn: Optional[types.FINALIZE_FN] = None,
        add_X_imus: bool = False,
        add_X_imus_kwargs: dict = dict(),
        add_X_jointaxes: bool = False,
        add_X_jointaxes_kwargs: dict = dict(),
        add_y_relpose: bool = False,
        add_y_rootincl: bool = False,
        add_y_rootincl_kwargs: dict = dict(),
        add_y_rootfull: bool = False,
        add_y_rootfull_kwargs: dict = dict(),
        sys_ml: Optional[base.System] = None,
        randomize_positions: bool = False,
        randomize_motion_artifacts: bool = False,
        randomize_joint_params: bool = False,
        randomize_hz: bool = False,
        randomize_hz_kwargs: dict = dict(),
        imu_motion_artifacts: bool = False,
        imu_motion_artifacts_kwargs: dict = dict(),
        dynamic_simulation: bool = False,
        dynamic_simulation_kwargs: dict = dict(),
        output_transform: Optional[Callable] = None,
        keep_output_extras: bool = False,
        use_link_number_in_Xy: bool = False,
        cor: bool = False,
        disable_tqdm: bool = False,
    ) -> None:
        "Random Chain Motion Generator"

        # add some default values
        randomize_hz_kwargs_defaults = dict(add_dt=True)
        randomize_hz_kwargs_defaults.update(randomize_hz_kwargs)
        randomize_hz_kwargs = randomize_hz_kwargs_defaults

        sys, config = utils.to_list(sys), utils.to_list(config)
        sys_ml = sys[0] if sys_ml is None else sys_ml

        for c in config:
            assert c.is_feasible()

        self.gens = []
        for _sys in sys:
            self.gens.append(
                _build_mconfig_batched_generator(
                    sys=_sys,
                    config=config,
                    setup_fn=setup_fn,
                    finalize_fn=finalize_fn,
                    add_X_imus=add_X_imus,
                    add_X_imus_kwargs=add_X_imus_kwargs,
                    add_X_jointaxes=add_X_jointaxes,
                    add_X_jointaxes_kwargs=add_X_jointaxes_kwargs,
                    add_y_relpose=add_y_relpose,
                    add_y_rootincl=add_y_rootincl,
                    add_y_rootincl_kwargs=add_y_rootincl_kwargs,
                    add_y_rootfull=add_y_rootfull,
                    add_y_rootfull_kwargs=add_y_rootfull_kwargs,
                    sys_ml=sys_ml,
                    randomize_positions=randomize_positions,
                    randomize_motion_artifacts=randomize_motion_artifacts,
                    randomize_joint_params=randomize_joint_params,
                    randomize_hz=randomize_hz,
                    randomize_hz_kwargs=randomize_hz_kwargs,
                    imu_motion_artifacts=imu_motion_artifacts,
                    imu_motion_artifacts_kwargs=imu_motion_artifacts_kwargs,
                    dynamic_simulation=dynamic_simulation,
                    dynamic_simulation_kwargs=dynamic_simulation_kwargs,
                    output_transform=output_transform,
                    keep_output_extras=keep_output_extras,
                    use_link_number_in_Xy=use_link_number_in_Xy,
                    cor=cor,
                )
            )

        self._n_mconfigs = len(config)
        self._size_of_generators = [self._n_mconfigs] * len(self.gens)

        self._disable_tqdm = disable_tqdm

    def _compute_repeats(self, sizes: int | list[int]) -> list[int]:
        "how many times the generators are repeated to create a batch of `sizes`"

        S, L = sum(self._size_of_generators), len(self._size_of_generators)

        def assert_size(size: int):
            assert self._n_mconfigs in utils.primes(size), (
                f"`size`={size} is not divisible by number of "
                + f"`mconfigs`={self._n_mconfigs}"
            )

        if isinstance(sizes, int):
            assert (sizes // S) > 0, f"Batchsize or size too small. {sizes} < {S}"
            assert sizes % S == 0, f"`size`={sizes} not divisible by {S}"
            repeats = L * [sizes // S]
        else:
            for size in sizes:
                assert_size(size)

            assert len(sizes) == len(
                self.gens
            ), f"len(`sizes`)={len(sizes)} != {len(self.gens)}"

            repeats = [
                size // size_of_gen
                for size, size_of_gen in zip(sizes, self._size_of_generators)
            ]
            assert 0 not in repeats

        return repeats

    def to_lazy_gen(
        self, sizes: int | list[int] = 1, jit: bool = True
    ) -> types.BatchedGenerator:
        return batch.generators_lazy(self.gens, self._compute_repeats(sizes), jit)

    @staticmethod
    def _number_of_executions_required(size: int) -> int:
        _, vmap = utils.distribute_batchsize(size)

        eager_threshold = utils.batchsize_thresholds()[1]
        primes = iter(utils.primes(vmap))
        n_calls = 1
        while vmap > eager_threshold:
            prime = next(primes)
            n_calls *= prime
            vmap /= prime

        return n_calls

    def _generators_ncalls(self, sizes: int | list[int] = 1):
        "Returns list of unbatched sequences as numpy arrays."
        repeats = self._compute_repeats(sizes)
        sizes = list(jnp.array(repeats) * jnp.array(self._size_of_generators))

        reduced_repeats = []
        n_calls = []
        for size, repeat in zip(sizes, repeats):
            n_call = self._number_of_executions_required(size)
            gcd = utils.gcd(n_call, repeat)
            n_calls.append(gcd)
            reduced_repeats.append(repeat // gcd)
        jits = [N > 1 for N in n_calls]

        gens = []
        for i in range(len(repeats)):
            gens.append(
                batch.generators_lazy([self.gens[i]], [reduced_repeats[i]], jits[i])
            )

        return gens, n_calls

    def to_list(
        self, sizes: int | list[int] = 1, seed: int = 1
    ) -> list[tree_utils.PyTree[np.ndarray]]:
        "Returns list of unbatched sequences as numpy arrays."
        gens, n_calls = self._generators_ncalls(sizes)

        data = []
        batch.generators_eager(
            gens, n_calls, lambda d: data.extend(d), seed, self._disable_tqdm
        )
        return data

    def to_folder(
        self,
        path: str,
        sizes: int | list[int] = 1,
        seed: int = 1,
        overwrite: bool = True,
        file_prefix: str = "seq",
        save_fn: Callable[[PyTree[np.ndarray], str], None] = partial(
            utils.pickle_save, overwrite=True
        ),
        verbose: bool = True,
    ):

        i = 0

        def callback(data: list[PyTree[np.ndarray]]) -> None:
            nonlocal i
            data = utils.replace_elements_w_nans(data, verbose=verbose)
            for d in data:
                file = utils.parse_path(
                    path, file_prefix + str(i), file_exists_ok=overwrite
                )
                save_fn(d, file)
                i += 1
            # cleanup
            del data

        gens, n_calls = self._generators_ncalls(sizes)
        batch.generators_eager(gens, n_calls, callback, seed, self._disable_tqdm)

    def to_pickle(
        self,
        path: str,
        sizes: int | list[int] = 1,
        seed: int = 1,
        overwrite: bool = True,
    ) -> None:
        data = tree_utils.tree_batch(self.to_list(sizes, seed), backend="numpy")
        utils.pickle_save(data, path, overwrite=overwrite)

    def to_eager_gen(
        self,
        batchsize: int = 1,
        sizes: int | list[int] = 1,
        seed: int = 1,
        shuffle: bool = True,
        transform=None,
    ) -> types.BatchedGenerator:
        data = self.to_list(sizes, seed)
        assert len(data) >= batchsize
        return self.eager_gen_from_list(data, batchsize, shuffle, transform)

    @staticmethod
    def eager_gen_from_list(
        data: list[tree_utils.PyTree],
        batchsize: int,
        shuffle: bool = True,
        transform=None,
    ) -> types.BatchedGenerator:
        data = data.copy()
        n_batches, i = len(data) // batchsize, 0

        def generator(key: jax.Array):
            nonlocal i
            if shuffle and i == 0:
                random.shuffle(data)

            start, stop = i * batchsize, (i + 1) * batchsize
            batch = tree_utils.tree_batch(data[start:stop], backend="numpy")
            batch = utils.pytree_deepcopy(batch)
            if transform is not None:
                batch = transform(batch)

            i = (i + 1) % n_batches
            return batch

        return generator


def _copy_dicts(f) -> dict:
    def _f(*args, **kwargs):
        _copy = lambda obj: obj.copy() if isinstance(obj, dict) else obj
        args = tuple([_copy(ele) for ele in args])
        kwargs = {k: _copy(v) for k, v in kwargs.items()}
        return f(*args, **kwargs)

    return _f


@_copy_dicts
def _build_mconfig_batched_generator(
    sys: base.System,
    config: list[jcalc.MotionConfig],
    setup_fn: types.SETUP_FN | None,
    finalize_fn: types.FINALIZE_FN | None,
    add_X_imus: bool,
    add_X_imus_kwargs: dict,
    add_X_jointaxes: bool,
    add_X_jointaxes_kwargs: dict,
    add_y_relpose: bool,
    add_y_rootincl: bool,
    add_y_rootincl_kwargs: dict,
    add_y_rootfull: bool,
    add_y_rootfull_kwargs: dict,
    sys_ml: base.System,
    randomize_positions: bool,
    randomize_motion_artifacts: bool,
    randomize_joint_params: bool,
    randomize_hz: bool,
    randomize_hz_kwargs: dict,
    imu_motion_artifacts: bool,
    imu_motion_artifacts_kwargs: dict,
    dynamic_simulation: bool,
    dynamic_simulation_kwargs: dict,
    output_transform: Callable | None,
    keep_output_extras: bool,
    use_link_number_in_Xy: bool,
    cor: bool,
) -> types.BatchedGenerator:

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

        if "prob_rigid" in imu_motion_artifacts_kwargs:
            assert randomize_motion_artifacts, (
                "`prob_rigid` works by overwriting damping and stiffness parameters "
                "using the `randomize_motion_artifacts` flag, so it must be enabled."
            )

    if dynamic_simulation:
        finalize_fns.DynamicalSimulation.assert_test_system(sys)

    def _setup_fn(key: types.PRNGKey, sys: base.System) -> base.System:
        pipe = []
        if imu_motion_artifacts and randomize_motion_artifacts:
            pipe.append(
                motion_artifacts.setup_fn_randomize_damping_stiffness_factory(
                    **imu_motion_artifacts_kwargs
                )
            )
        if randomize_positions:
            pipe.append(setup_fns._setup_fn_randomize_positions)
        if randomize_joint_params:
            pipe.append(jcalc._init_joint_params)
        if setup_fn is not None:
            pipe.append(setup_fn)

        for f in pipe:
            key, consume = jax.random.split(key)
            sys = f(consume, sys)
        if cor:
            sys = sys._replace_free_with_cor()
        return sys

    def _finalize_fn(Xy: types.Xy, extras: types.OutputExtras):
        pipe = []
        if dynamic_simulation:
            pipe.append(finalize_fns.DynamicalSimulation(**dynamic_simulation_kwargs))
        if imu_motion_artifacts and imu_motion_artifacts_kwargs.get(
            "hide_injected_bodies", True
        ):
            pipe.append(motion_artifacts.HideInjectedBodies())
        if finalize_fn is not None:
            pipe.append(finalize_fns.FinalizeFn(finalize_fn))
        if add_X_imus:
            pipe.append(finalize_fns.IMU(**add_X_imus_kwargs))
        if add_X_jointaxes:
            pipe.append(
                finalize_fns.JointAxisSensor(sys_noimu, **add_X_jointaxes_kwargs)
            )
        if add_y_relpose:
            pipe.append(finalize_fns.RelPose(sys_noimu))
        if add_y_rootincl:
            assert not add_y_rootfull
            pipe.append(finalize_fns.RootIncl(sys_noimu, **add_y_rootincl_kwargs))
        if add_y_rootfull:
            assert not add_y_rootincl
            pipe.append(finalize_fns.RootFull(sys_noimu, **add_y_rootfull_kwargs))
        if use_link_number_in_Xy:
            pipe.append(finalize_fns.Names2Indices(sys_noimu))

        for f in pipe:
            Xy, extras = f(Xy, extras)
        return Xy, extras

    def _gen(key: types.PRNGKey):
        key, *consume = jax.random.split(key, len(config) + 1)
        syss = jax.vmap(_setup_fn, (0, None))(jnp.array(consume), sys)

        if randomize_hz:
            assert "sampling_rates" in randomize_hz_kwargs
            hzs = randomize_hz_kwargs["sampling_rates"]
            assert len(set([c.T for c in config])) == 1
            N = int(min(hzs) * config[0].T)
            key, consume = jax.random.split(key)
            dt = 1 / jax.random.choice(consume, jnp.array(hzs))
            # makes sys.dt from float to AbstractArray
            syss = syss.replace(dt=jnp.array(dt))
        else:
            N = None

        qs = []
        for i, _config in enumerate(config):
            key, _q = draw_random_q(key, syss[i], _config, N)
            qs.append(_q)
        qs = jnp.stack(qs)

        @jax.vmap
        def _vmapped_context(key, q, sys):
            x, _ = jax.vmap(kinematics.forward_kinematics_transforms, (None, 0))(sys, q)
            X = (
                {"dt": jnp.array(sys.dt)}
                if (randomize_hz and randomize_hz_kwargs["add_dt"])
                else {}
            )
            Xy, extras = (X, {}), (key, q, x, sys)
            return _finalize_fn(Xy, extras)

        keys = jax.random.split(key, len(config))
        Xy, extras = _vmapped_context(keys, qs, syss)
        output = (Xy, extras) if keep_output_extras else Xy
        output = output if output_transform is None else output_transform(output)
        return output

    return _gen


def draw_random_q(
    key: types.PRNGKey,
    sys: base.System,
    config: jcalc.MotionConfig,
    N: int | None,
) -> tuple[types.Xy, types.OutputExtras]:

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

        if link_type in config.joint_type_specific_overwrites:
            _config = replace(
                config, **config.joint_type_specific_overwrites[link_type]
            )
        else:
            _config = config

        q_link = draw_fn(_config, key_t, key_value, sys.dt, N, joint_params)
        # even revolute and prismatic joints must be 2d arrays
        q_link = q_link if q_link.ndim == 2 else q_link[:, None]
        q_list.append(q_link)
        return key

    keys = sys.scan(draw_q, "ll", sys.link_types, sys.links)
    # stack of keys; only the last key is unused
    key = keys[-1]

    q = jnp.concatenate(q_list, axis=1)

    return key, q
