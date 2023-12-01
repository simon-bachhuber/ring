import warnings

import jax
import jax.numpy as jnp
import tree_utils

from ... import base
from ...io import load_sys_from_str
from ...io import parse_system
from ...io import save_sys_to_str
from ...scan import scan_sys


def imu_reference_link_name(imu_link_name: str) -> str:
    return "_" + imu_link_name


def unactuated_subsystem(sys) -> list[str]:
    return [imu_reference_link_name(name) for name in sys.findall_imus()]


def _subsystem_factory(imu_name: str, pos_min_max: float) -> base.System:
    assert pos_min_max >= 0
    pos = f'pos_min="-{pos_min_max} -{pos_min_max} -{pos_min_max}" pos_max="{pos_min_max} {pos_min_max} {pos_min_max}"'  # noqa: E501
    stiff = 'spring_stiff="50 50 50"'
    damping = 'damping="5 5 5"'
    return load_sys_from_str(
        f"""
        <x_xy>
        <worldbody>
        <body name="{imu_name}" joint="p3d" {pos if pos_min_max != 0.0 else ""} {stiff} {damping}/>
        </worldbody>
        </x_xy>
        """  # noqa: E501
    )


def inject_subsystems(
    sys: base.System, pos_min_max: float = 0.0, **kwargs
) -> base.System:
    from x_xy.subpkgs import sys_composer

    imu_idx_to_name_map = {sys.name_to_idx(imu): imu for imu in sys.findall_imus()}

    default_spher_stif = jnp.ones((3,)) * 0.3
    default_spher_damp = default_spher_stif * 0.1
    for imu in sys.findall_imus():
        sys = sys.unfreeze(imu, "spherical")
        # set default stiffness and damping of spherical joint
        # this won't override anything because the frozen joint can not have any values
        qd_slice = sys.idx_map("d")[imu]
        stiffne = sys.link_spring_stiffness.at[qd_slice].set(default_spher_stif)
        damping = sys.link_damping.at[qd_slice].set(default_spher_damp)
        sys = sys.replace(link_spring_stiffness=stiffne, link_damping=damping)

        _imu = imu_reference_link_name(imu)
        sys = sys.rename_link(imu, _imu)
        sys = sys_composer.inject_system(
            sys, _subsystem_factory(imu, pos_min_max), _imu
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

    # re-calculate the inertia matrices because the geoms have been re-attached
    sys = parse_system(sys)

    # TODO set all joint_params to zeros; they can not be preserved anyways and
    # otherwise many warnings will be rose
    # instead warn explicitly once now and move on
    warnings.warn(
        "`sys.links.joint_params` has been set to zero, this might lead to "
        "unexpected behaviour unless you use `randomize_joint_params`"
    )
    joint_params_zeros = tree_utils.tree_zeros_like(sys.links.joint_params)
    sys = sys.replace(links=sys.links.replace(joint_params=joint_params_zeros))

    # double load; this fixes the issue that injected links got appended at the end
    sys = load_sys_from_str(save_sys_to_str(sys))

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


def setup_fn_randomize_damping_stiffness_factory(prob_rigid: float):
    assert 0 <= prob_rigid <= 1
    assert prob_rigid != 1, "Use `imu_motion_artifacts`=False instead."

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
        for imu in sys.findall_imus():
            # _imu has spherical joint and imu has p3d joint
            slice = jnp.r_[idx_map[imu_reference_link_name(imu)], idx_map[imu]]
            key, c1, c2 = jax.random.split(key, 3)
            if prob_rigid > 0:
                is_rigid = jax.random.bernoulli(c1, prob_rigid)
                stif, damp = jax.lax.cond(
                    is_rigid, stif_damp_rigid, stif_damp_nonrigid, c2
                )
            else:
                stif, damp = stif_damp_nonrigid(c2)
            link_spring_stiffness = link_spring_stiffness.at[slice].set(stif)
            link_damping = link_damping.at[slice].set(damp)

        return sys.replace(
            link_damping=link_damping, link_spring_stiffness=link_spring_stiffness
        )

    return setup_fn_randomize_damping_stiffness


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

    scan_sys(sys_small, f, "l", sys_small.link_names)

    x_small = tree_utils.tree_indices(x_large, jnp.array(x_small_indices), axis=1)
    q_small = jnp.concatenate(q_small, axis=1)
    return q_small, x_small


class GeneratorTrafoHideInjectedBodies:
    def __call__(self, gen):
        from x_xy.subpkgs import sys_composer

        def _gen(*args):
            (X, y), (key, q, x, sys_x) = gen(*args)

            # delete injected frames; then rename from `_imu` back to `imu`
            imus = sys_x.findall_imus()
            _imu2imu_map = {imu_reference_link_name(imu): imu for imu in imus}
            sys = sys_composer.delete_subsystem(sys_x, imus)
            for _imu, imu in _imu2imu_map.items():
                sys = sys.rename_link(_imu, imu).change_joint_type(imu, "frozen")

            # match q and x to `sys`; second axis is link axis
            q, x = _match_q_x_between_sys(sys, q, x, sys_x, q_large_skip=imus)

            return (X, y), (key, q, x, sys)

        return _gen