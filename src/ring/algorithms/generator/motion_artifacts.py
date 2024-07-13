import inspect
import warnings

import jax
import jax.numpy as jnp
import tree_utils

from ring import base
from ring import io


def imu_reference_link_name(imu_link_name: str) -> str:
    return "_" + imu_link_name


def unactuated_subsystem(sys) -> list[str]:
    return [imu_reference_link_name(name) for name in sys.findall_imus()]


def _subsystem_factory(
    imu_name: str,
    pos_min_max: float,
    translational_stif: float,
    translational_damp: float,
) -> base.System:
    assert pos_min_max >= 0
    pos = f'pos_min="-{pos_min_max} -{pos_min_max} -{pos_min_max}" pos_max="{pos_min_max} {pos_min_max} {pos_min_max}"'  # noqa: E501
    stiff = (
        f'spring_stiff="{translational_stif} {translational_stif} {translational_stif}"'
    )
    translational_damp = translational_stif * translational_damp
    damping = (
        f'damping="{translational_damp} {translational_damp} {translational_damp}"'
    )
    return io.load_sys_from_str(
        f"""
        <x_xy>
        <worldbody>
        <body name="{imu_name}" joint="p3d" {pos if pos_min_max != 0.0 else ""} {stiff} {damping}/>
        </worldbody>
        </x_xy>
        """  # noqa: E501
    )


def inject_subsystems(
    sys: base.System,
    pos_min_max: float = 0.0,
    rotational_stif: float = 0.3,
    rotational_damp: float = 0.1,
    translational_stif: float = 50.0,
    translational_damp: float = 0.1,
    disable_warning: bool = False,
    **kwargs,  # needed because `imu_motion_artifacts_kwargs` is used
    # for `setup_fn_randomize_damping_stiffness_factory` also
) -> base.System:
    imu_idx_to_name_map = {sys.name_to_idx(imu): imu for imu in sys.findall_imus()}

    default_spher_stif = jnp.ones((3,)) * rotational_stif
    default_spher_damp = default_spher_stif * rotational_damp
    for imu in sys.findall_imus():
        sys = sys.unfreeze(imu, "spherical")
        # set default stiffness and damping of spherical joint
        # this won't override anything because the frozen joint can not have any values
        qd_slice = sys.idx_map("d")[imu]
        stiffne = sys.link_spring_stiffness.at[qd_slice].set(default_spher_stif)
        damping = sys.link_damping.at[qd_slice].set(default_spher_damp)
        sys = sys.replace(link_spring_stiffness=stiffne, link_damping=damping)

        _imu = imu_reference_link_name(imu)
        sys = sys.change_link_name(imu, _imu)
        sys = sys.inject_system(
            _subsystem_factory(
                imu, pos_min_max, translational_stif, translational_damp
            ),
            _imu,
        )

    # attach geoms to newly injected link
    new_geoms = []

    for geom in sys.geoms:
        if geom.link_idx in imu_idx_to_name_map:
            imu_name = imu_idx_to_name_map[geom.link_idx]
            new_link_idx = sys.name_to_idx(imu_name)
            geom = geom.replace(link_idx=new_link_idx)
        new_geoms.append(geom)

    sys = sys.replace(geoms=new_geoms)

    # TODO investigate whether this parse is needed; I don't think so
    # re-calculate the inertia matrices because the geoms have been re-attached
    sys = sys.parse()

    # TODO set all joint_params to zeros; they can not be preserved anyways and
    # otherwise many warnings will be rose
    # instead warn explicitly once now and move on
    if not disable_warning:
        warnings.warn(
            "`sys.links.joint_params` has been set to zero, this might lead to "
            "unexpected behaviour unless you use `randomize_joint_params`"
        )
    joint_params_zeros = tree_utils.tree_zeros_like(sys.links.joint_params)
    sys = sys.replace(links=sys.links.replace(joint_params=joint_params_zeros))

    # double load; this fixes the issue that injected links got appended at the end
    sys = io.load_sys_from_str(io.save_sys_to_str(sys))

    return sys


_STIF_MIN_SPH = 0.2
_STIF_MAX_SPH = 10.0
_STIF_MIN_P3D = 25.0
_STIF_MAX_P3D = 1e3
# damping = factor * stiffness
_DAMP_MIN = 0.05
_DAMP_MAX = 0.5


def _log_uniform(key, shape, minval, maxval):
    assert 0 <= minval <= maxval
    minval, maxval = map(jnp.log, (minval, maxval))
    return jnp.exp(jax.random.uniform(key, shape, minval=minval, maxval=maxval))


def setup_fn_randomize_damping_stiffness_factory(
    prob_rigid: float = 0.0,
    all_imus_either_rigid_or_flex: bool = False,
    imus_surely_rigid: list[str] = [],
    **kwargs,
):
    assert 0 <= prob_rigid <= 1
    assert prob_rigid != 1, "Use `imu_motion_artifacts`=False instead."
    if prob_rigid == 0.0:
        assert len(imus_surely_rigid) == 0

    def stif_damp_rigid(key):
        stif_sph = 200.0 * jnp.ones((3,))
        stif_p3d = 2e4 * jnp.ones((3,))
        stif = jnp.concatenate((stif_sph, stif_p3d))
        return stif, stif * 0.2

    def stif_damp_nonrigid(key):
        keys = jax.random.split(key, 3)
        stif_sph = _log_uniform(keys[0], (3,), _STIF_MIN_SPH, _STIF_MAX_SPH)
        stif_p3d = _log_uniform(keys[1], (3,), _STIF_MIN_P3D, _STIF_MAX_P3D)
        stif = jnp.concatenate((stif_sph, stif_p3d))
        damp = _log_uniform(keys[2], (6,), _DAMP_MIN, _DAMP_MAX)
        return stif, stif * damp

    def setup_fn_randomize_damping_stiffness(key, sys: base.System) -> base.System:
        link_damping = sys.link_damping
        link_spring_stiffness = sys.link_spring_stiffness

        idx_map = sys.idx_map("d")
        imus = sys.findall_imus()

        # initialize this RV because it might not get redrawn if
        # `all_imus_either_rigid_or_flex` is set
        key, consume = jax.random.split(key)
        is_rigid = jax.random.bernoulli(consume, prob_rigid)

        # this is only for the assertion used below
        triggered_surely_rigid = []

        for imu in imus:
            # _imu has spherical joint and imu has p3d joint
            slice = jnp.r_[idx_map[imu_reference_link_name(imu)], idx_map[imu]]
            key, c1, c2 = jax.random.split(key, 3)

            if prob_rigid > 0:
                if imu in imus_surely_rigid:
                    triggered_surely_rigid.append(imu)
                    # logging.debug(f"IMU {imu} is surely rigid.")
                    stif, damp = stif_damp_rigid(c2)
                else:
                    if not all_imus_either_rigid_or_flex:
                        is_rigid = jax.random.bernoulli(c1, prob_rigid)
                    stif, damp = jax.lax.cond(
                        is_rigid, stif_damp_rigid, stif_damp_nonrigid, c2
                    )
            else:
                stif, damp = stif_damp_nonrigid(c2)
            link_spring_stiffness = link_spring_stiffness.at[slice].set(stif)
            link_damping = link_damping.at[slice].set(damp)

        assert len(imus_surely_rigid) == len(
            triggered_surely_rigid
        ), f"{imus_surely_rigid}, {triggered_surely_rigid}"
        for imu_surely_rigid in imus_surely_rigid:
            assert (
                imu_surely_rigid in triggered_surely_rigid
            ), f"{imus_surely_rigid} not in {triggered_surely_rigid}"

        return sys.replace(
            link_damping=link_damping, link_spring_stiffness=link_spring_stiffness
        )

    return setup_fn_randomize_damping_stiffness


# assert that there exists no keyword arg duplicate which would induce ambiguity
kwargs = lambda f: set(inspect.signature(f).parameters.keys())
assert (
    len(
        kwargs(inject_subsystems).intersection(
            kwargs(setup_fn_randomize_damping_stiffness_factory)
        )
    )
    == 1
)


def _match_q_x_between_sys(
    sys_small: base.System,
    q_large: jax.Array,
    x_large: base.Transform,
    sys_large: base.System,
    q_large_skip: list[str],
) -> tree_utils.PyTree:
    assert q_large.ndim == 2
    assert q_large.shape[1] == sys_large.q_size()
    assert x_large.shape(1) == sys_large.num_links()

    x_small_indices = []
    q_small = []
    q_idx_map = sys_large.idx_map("q")

    def f(_, __, name: str):
        x_small_indices.append(sys_large.name_to_idx(name))
        # for the imu links the joint type was changed from spherical to frozen
        # thus the q_idx_map has slices of length 4 but the `sys_small` has those
        # imus but with frozen joint type and thus slices of length 0; so skip them
        if name in q_large_skip:
            return
        q_small.append(q_large[:, q_idx_map[name]])

    sys_small.scan(f, "l", sys_small.link_names)

    x_small = tree_utils.tree_indices(x_large, jnp.array(x_small_indices), axis=1)
    q_small = jnp.concatenate(q_small, axis=1)
    return q_small, x_small


class HideInjectedBodies:
    def __call__(self, Xy, extras):
        (X, y), (key, q, x, sys_x) = Xy, extras

        # delete injected frames; then rename from `_imu` back to `imu`
        imus = sys_x.findall_imus()
        _imu2imu_map = {imu_reference_link_name(imu): imu for imu in imus}
        sys = sys_x.delete_system(imus)
        for _imu, imu in _imu2imu_map.items():
            sys = sys.change_link_name(_imu, imu).change_joint_type(imu, "frozen")

        # match q and x to `sys`; second axis is link axis
        q, x = _match_q_x_between_sys(sys, q, x, sys_x, q_large_skip=imus)

        return (X, y), (key, q, x, sys)
