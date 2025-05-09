from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import Any, Callable, get_type_hints, Optional

import jax
import jax.numpy as jnp
import tree_utils

from ring import algebra
from ring import base
from ring import maths
from ring.algorithms import _random
from ring.algorithms._random import _to_float
from ring.algorithms._random import TimeDependentFloat


@dataclass
class MotionConfig:
    """
    Configuration for joint motion generation in kinematic and dynamic simulations.

    This class defines the constraints and parameters for generating random joint motions,
    including angular and positional velocity limits, interpolation methods, and range
    restrictions for various joint types.

    Attributes:
        T (float): Total duration of the motion sequence (in seconds).
        t_min (float): Minimum time interval between two generated joint states.
        t_max (float | TimeDependentFloat): Maximum time interval between two generated joint states.

        dang_min (float | TimeDependentFloat): Minimum angular velocity (rad/s).
        dang_max (float | TimeDependentFloat): Maximum angular velocity (rad/s).
        dang_min_free_spherical (float | TimeDependentFloat): Minimum angular velocity for free and spherical joints.
        dang_max_free_spherical (float | TimeDependentFloat): Maximum angular velocity for free and spherical joints.

        delta_ang_min (float | TimeDependentFloat): Minimum allowed change in joint angle (radians).
        delta_ang_max (float | TimeDependentFloat): Maximum allowed change in joint angle (radians).
        delta_ang_min_free_spherical (float | TimeDependentFloat): Minimum allowed change in angle for free/spherical joints.
        delta_ang_max_free_spherical (float | TimeDependentFloat): Maximum allowed change in angle for free/spherical joints.

        dpos_min (float | TimeDependentFloat): Minimum translational velocity.
        dpos_max (float | TimeDependentFloat): Maximum translational velocity.
        pos_min (float | TimeDependentFloat): Minimum position constraint.
        pos_max (float | TimeDependentFloat): Maximum position constraint.

        pos_min_p3d_x (float | TimeDependentFloat): Minimum position in x-direction for P3D joints.
        pos_max_p3d_x (float | TimeDependentFloat): Maximum position in x-direction for P3D joints.
        pos_min_p3d_y (float | TimeDependentFloat): Minimum position in y-direction for P3D joints.
        pos_max_p3d_y (float | TimeDependentFloat): Maximum position in y-direction for P3D joints.
        pos_min_p3d_z (float | TimeDependentFloat): Minimum position in z-direction for P3D joints.
        pos_max_p3d_z (float | TimeDependentFloat): Maximum position in z-direction for P3D joints.

        cdf_bins_min (int): Minimum number of bins for cumulative distribution function (CDF)-based random sampling.
        cdf_bins_max (Optional[int]): Maximum number of bins for CDF-based sampling.

        randomized_interpolation_angle (bool): Whether to use randomized interpolation for angular motion.
        randomized_interpolation_position (bool): Whether to use randomized interpolation for positional motion.
        interpolation_method (str): Interpolation method to be used (default: "cosine").

        range_of_motion_hinge (bool): Whether to enforce range-of-motion constraints on hinge joints.
        range_of_motion_hinge_method (str): Method used for range-of-motion enforcement (e.g., "uniform", "sigmoid").

        rom_halfsize (float | TimeDependentFloat): Half-size of the range of motion restriction.

        ang0_min (float): Minimum initial joint angle.
        ang0_max (float): Maximum initial joint angle.
        pos0_min (float): Minimum initial joint position.
        pos0_max (float): Maximum initial joint position.

        cor_t_min (float): Minimum time step for center-of-rotation (COR) joints.
        cor_t_max (float | TimeDependentFloat): Maximum time step for COR joints.
        cor_dpos_min (float | TimeDependentFloat): Minimum velocity for COR translation.
        cor_dpos_max (float | TimeDependentFloat): Maximum velocity for COR translation.
        cor_pos_min (float | TimeDependentFloat): Minimum position for COR translation.
        cor_pos_max (float | TimeDependentFloat): Maximum position for COR translation.
        cor_pos0_min (float): Initial minimum position for COR translation.
        cor_pos0_max (float): Initial maximum position for COR translation.

        joint_type_specific_overwrites (dict[str, dict[str, Any]]):
            A dictionary mapping joint types to specific motion configuration overrides.

    Methods:
        is_feasible:
            Checks if the motion configuration satisfies all constraints.

        to_nomotion_config:
            Returns a new `MotionConfig` where all velocities and angle changes are set to zero.

        overwrite_for_joint_type:
            Applies specific configuration changes for a given joint type.
            Note: These changes affect all instances of `MotionConfig` for this joint type.

        overwrite_for_subsystem:
            Modifies the motion configuration for all joints in a subsystem rooted at `link_name`.

        from_register:
            Retrieves a predefined `MotionConfig` from the global registry.
    """  # noqa: E501

    T: float = 60.0  # length of random motion
    t_min: float = 0.05  # min time between two generated angles
    t_max: float | TimeDependentFloat = 0.30  # max time ..

    dang_min: float | TimeDependentFloat = 0.1  # minimum angular velocity in rad/s
    dang_max: float | TimeDependentFloat = 3.0  # maximum angular velocity in rad/s

    # minimum angular velocity of euler angles used for `free and spherical joints`
    dang_min_free_spherical: float | TimeDependentFloat = 0.1
    dang_max_free_spherical: float | TimeDependentFloat = 3.0

    # max min allowed actual delta values in radians
    delta_ang_min: float | TimeDependentFloat = 0.0
    delta_ang_max: float | TimeDependentFloat = 2 * jnp.pi
    delta_ang_min_free_spherical: float | TimeDependentFloat = 0.0
    delta_ang_max_free_spherical: float | TimeDependentFloat = 2 * jnp.pi

    dpos_min: float | TimeDependentFloat = 0.001  # speed of translation
    dpos_max: float | TimeDependentFloat = 0.7
    pos_min: float | TimeDependentFloat = -2.5
    pos_max: float | TimeDependentFloat = +2.5
    pos_min_p3d_x: float | TimeDependentFloat = -2.5
    pos_max_p3d_x: float | TimeDependentFloat = +2.5
    pos_min_p3d_y: float | TimeDependentFloat = -2.5
    pos_max_p3d_y: float | TimeDependentFloat = +2.5
    pos_min_p3d_z: float | TimeDependentFloat = -2.5
    pos_max_p3d_z: float | TimeDependentFloat = +2.5

    # used by both `random_angle_*` and `random_pos_*`
    # only used if `randomized_interpolation` is set
    cdf_bins_min: int = 5
    # by default equal to `cdf_bins_min`
    cdf_bins_max: Optional[int] = None

    # flags
    randomized_interpolation_angle: bool = False
    randomized_interpolation_position: bool = False
    interpolation_method: str = "cosine"
    range_of_motion_hinge: bool = True
    range_of_motion_hinge_method: str = "uniform"

    # this value has nothing to do with `range_of_motion` flag
    # this forces the value to stay within [ANG_0 - rom_halfsize, ANG_0 + rom_halfsize]
    # used only by the `_draw_rxyz` function
    rom_halfsize: float | TimeDependentFloat = 2 * jnp.pi

    # initial value of joints
    ang0_min: float = -jnp.pi
    ang0_max: float = jnp.pi
    pos0_min: float = 0.0
    pos0_max: float = 0.0
    pos0_min_p3d_x: float = 0.0
    pos0_max_p3d_x: float = 0.0
    pos0_min_p3d_y: float = 0.0
    pos0_max_p3d_y: float = 0.0
    pos0_min_p3d_z: float = 0.0
    pos0_max_p3d_z: float = 0.0

    # cor (center of rotation) custom fields
    cor_t_min: float = 0.2
    cor_t_max: float | TimeDependentFloat = 2.0
    cor_dpos_min: float | TimeDependentFloat = 0.00001
    cor_dpos_max: float | TimeDependentFloat = 0.5
    cor_pos_min: float | TimeDependentFloat = -0.4
    cor_pos_max: float | TimeDependentFloat = 0.4
    cor_pos0_min: float = 0.0
    cor_pos0_max: float = 0.0

    # specify changes for this motionconfig and for specific joint types
    # map of `link_types` -> dictionary of changes
    joint_type_specific_overwrites: dict[str, dict[str, Any]] = field(
        default_factory=lambda: dict()
    )

    # fields related to simulating standstills (no motion time periods)
    # these are "Joint Standstills" so the standstills are calculated on
    # a joint level, for each joint independently
    # This means that a `standstills_prob` of 20% means that each joint
    # has at each dt \in [t_min, t_max] drawing process a probability of
    # 20% that it will just stay at its current joint value
    include_standstills_prob: float = 0.0  # in %; 0% means no standstills
    include_standstills_t_min: float = 0.5
    include_standstills_t_max: float = 5.0

    def is_feasible(self) -> bool:
        return _is_feasible_config1(self)

    def to_nomotion_config(self) -> "MotionConfig":
        kwargs = asdict(self)
        for key in [
            "dang_min",
            "dang_max",
            "delta_ang_min",
            "dang_min_free_spherical",
            "dang_max_free_spherical",
            "delta_ang_min_free_spherical",
            "dpos_min",
            "dpos_max",
        ]:
            kwargs[key] = 0.0
        nomotion_config = MotionConfig(**kwargs)
        assert nomotion_config.is_feasible()
        return nomotion_config

    @staticmethod
    def overwrite_for_joint_type(joint_type: str, **changes) -> None:
        """Changes values of the `MotionConfig` used by the draw_fn for only a specific
        joint.
        !!! Note
            This applies these changes to *all* MotionConfigs for this joint type!
            This takes precedence *over* `Motionconfig.joint_type_specific_overwrites`!
        """
        previous_changes = _overwrite_for_joint_type_changes[joint_type]
        for change in changes:
            assert change not in previous_changes, f"For jointtype={joint_type} you "
            f"previously changed the value={change}. You can't change it again, this "
            "is not supported."
        previous_changes.update(changes)

        jm = get_joint_model(joint_type)

        def draw_fn(config, *args):
            return jm.rcmg_draw_fn(replace(config, **changes), *args)

        register_new_joint_type(
            joint_type,
            replace(jm, rcmg_draw_fn=draw_fn),
            base.Q_WIDTHS[joint_type],
            base.QD_WIDTHS[joint_type],
            overwrite=True,
        )

    @staticmethod
    def overwrite_for_subsystem(
        sys: base.System, link_name: str, **changes
    ) -> base.System:
        """Modifies motionconfig of all joints in subsystem with root `link_name`.
        Note that if the subsystem contains a free joint then the jointtype will
        will be re-named to `free_<link_name>`, then the RCMG flag `cor` will
        potentially not work as expected because it searches for all joints of
        type `free` to replace with `cor`. The workaround here is to change the
        type already from `free` to `cor in the xml file.
        This takes precedence *over* `Motionconfig.joint_type_specific_overwrites`!

        Args:
            sys (base.System): System object that gets updated
            link_name (str): Root node of subsystem
            changes: Changes to apply to the motionconfig

        Return:
            base.System: Updated system with new jointtypes
        """
        from ring.algorithms.generator.finalize_fns import _P_gains

        # all bodies in the subsystem
        bodies = sys.findall_bodies_subsystem(link_name) + [sys.name_to_idx(link_name)]

        jts_subsys = set([sys.link_types[i] for i in bodies]) - set(["frozen"])
        postfix = "_" + link_name
        # create new joint types with updated motionconfig
        for typ in jts_subsys:
            register_new_joint_type(
                typ + postfix,
                get_joint_model(typ),
                base.Q_WIDTHS[typ],
                base.QD_WIDTHS[typ],
            )
            MotionConfig.overwrite_for_joint_type(typ + postfix, **changes)
            _P_gains[typ + postfix] = _P_gains[typ]

        # rename all jointtypes
        new_link_types = [
            (
                sys.link_types[i] + postfix
                if (i in bodies and sys.link_types[i] != "frozen")
                else sys.link_types[i]
            )
            for i in range(sys.num_links())
        ]
        sys = sys.replace(link_types=new_link_types)
        return sys

    @staticmethod
    def from_register(name: str) -> "MotionConfig":
        return _registered_motion_configs[name]


_overwrite_for_joint_type_changes: dict[str, dict] = defaultdict(lambda: dict())


_registered_motion_configs = {
    "hinUndHer": MotionConfig(
        t_min=0.3,
        t_max=1.5,
        dang_max=3.0,
        delta_ang_min=0.5,
        pos_min=-1.5,
        pos_max=1.5,
        randomized_interpolation_angle=True,
    ),
    "langsam": MotionConfig(
        t_min=0.2,
        t_max=1.25,
        dang_max=2.0,
        randomized_interpolation_angle=True,
        dang_max_free_spherical=2.0,
        cdf_bins_min=1,
        cdf_bins_max=3,
        pos_min=-1.5,
        pos_max=1.5,
    ),
    "standard": MotionConfig(
        randomized_interpolation_angle=True,
        cdf_bins_min=1,
        cdf_bins_max=5,
    ),
    "expFast": MotionConfig(
        t_min=0.4,
        t_max=1.1,
        dang_max=jnp.deg2rad(180),
        delta_ang_min=jnp.deg2rad(60),
        delta_ang_max=jnp.deg2rad(110),
        pos_min=-1.5,
        pos_max=1.5,
        range_of_motion_hinge_method="sigmoid",
        randomized_interpolation_angle=True,
        cdf_bins_min=1,
        cdf_bins_max=3,
    ),
    "expSlow": MotionConfig(
        t_min=0.75,
        t_max=3.0,
        dang_min=0.1,
        dang_max=1.0,
        dang_min_free_spherical=0.1,
        delta_ang_min=0.4,
        dang_max_free_spherical=1.0,
        delta_ang_max_free_spherical=1.0,
        dpos_max=0.3,
        cor_dpos_max=0.3,
        range_of_motion_hinge_method="sigmoid",
        randomized_interpolation_angle=True,
        cdf_bins_min=1,
        cdf_bins_max=5,
    ),
    "expFastNoSig": MotionConfig(
        t_min=0.4,
        t_max=1.1,
        dang_max=jnp.deg2rad(180),
        delta_ang_min=jnp.deg2rad(60),
        delta_ang_max=jnp.deg2rad(110),
        pos_min=-1.5,
        pos_max=1.5,
        randomized_interpolation_angle=True,
        cdf_bins_min=1,
        cdf_bins_max=3,
    ),
    "expSlowNoSig": MotionConfig(
        t_min=0.75,
        t_max=3.0,
        dang_min=0.1,
        dang_max=1.0,
        dang_min_free_spherical=0.1,
        delta_ang_min=0.4,
        dang_max_free_spherical=1.0,
        delta_ang_max_free_spherical=1.0,
        dpos_max=0.3,
        cor_dpos_max=0.3,
        randomized_interpolation_angle=True,
        cdf_bins_min=1,
        cdf_bins_max=3,
    ),
    "verySlow": MotionConfig(
        t_min=1.5,
        t_max=5.0,
        dang_min=jnp.deg2rad(1),
        dang_max=jnp.deg2rad(30),
        delta_ang_min=jnp.deg2rad(20),
        dang_min_free_spherical=jnp.deg2rad(1),
        dang_max_free_spherical=jnp.deg2rad(10),
        delta_ang_min_free_spherical=jnp.deg2rad(5),
        dpos_max=0.3,
        cor_dpos_max=0.3,
        randomized_interpolation_angle=True,
        cdf_bins_min=1,
        cdf_bins_max=3,
    ),
}


def _joint_specific_overwrites_free_cor(
    id: str, dang: float, dpos: float
) -> MotionConfig:
    changes = dict(
        dang_max_free_spherical=dang,
        dpos_max=dpos,
        cor_dpos_max=dpos,
        t_min=1.5,
        t_max=15.0,
    )
    return replace(
        _registered_motion_configs[id],
        joint_type_specific_overwrites=dict(free=changes, cor=changes),
    )


_registered_motion_configs.update(
    {
        f"{id}-S": _joint_specific_overwrites_free_cor(id, 0.2, 0.1)
        for id in ["expSlow", "expFast", "hinUndHer", "standard"]
    }
)
_registered_motion_configs.update(
    {
        f"{id}-S+": _joint_specific_overwrites_free_cor(id, 0.1, 0.05)
        for id in ["expSlow", "expFast", "hinUndHer", "standard"]
    }
)
del _joint_specific_overwrites_free_cor


def _is_feasible_config1(c: MotionConfig) -> bool:
    t_min, t_max = c.t_min, _to_float(c.t_max, 0.0)

    def dx_deltax_check(dx_min, dx_max, deltax_min, deltax_max) -> bool:
        dx_min, dx_max, deltax_min, deltax_max = map(
            (lambda v: _to_float(v, 0.0)), (dx_min, dx_max, deltax_min, deltax_max)
        )
        if (deltax_max / t_min) < dx_min:
            return False
        if (deltax_min / t_max) > dx_max:
            return False
        return True

    cond1 = all(
        [
            dx_deltax_check(*args)
            for args in zip(
                [c.dang_min, c.dang_min_free_spherical],
                [c.dang_max, c.dang_max_free_spherical],
                [c.delta_ang_min, c.delta_ang_min_free_spherical],
                [c.delta_ang_max, c.delta_ang_max_free_spherical],
            )
        ]
    )

    # this one tests that the initial value is inside the feasible value range
    # so e.g. if you choose pos0_min=-10 then you can't choose pos_min=-1
    def inside_box_checks(x_min, x_max, x0_min, x0_max) -> bool:
        return (x0_min >= x_min) and (x0_max <= x_max)

    cond2 = inside_box_checks(
        _to_float(c.pos_min, 0.0), _to_float(c.pos_max, 0.0), c.pos0_min, c.pos0_max
    )
    cond3 = inside_box_checks(
        _to_float(c.pos_min_p3d_x, 0.0),
        _to_float(c.pos_max_p3d_x, 0.0),
        c.pos0_min_p3d_x,
        c.pos0_max_p3d_x,
    )
    cond4 = inside_box_checks(
        _to_float(c.pos_min_p3d_y, 0.0),
        _to_float(c.pos_max_p3d_y, 0.0),
        c.pos0_min_p3d_y,
        c.pos0_max_p3d_y,
    )
    cond5 = inside_box_checks(
        _to_float(c.pos_min_p3d_z, 0.0),
        _to_float(c.pos_max_p3d_z, 0.0),
        c.pos0_min_p3d_z,
        c.pos0_max_p3d_z,
    )

    # test that the delta_ang_min is smaller than 2*rom_halfsize
    cond6 = _to_float(c.delta_ang_min, 0.0) < 2 * _to_float(c.rom_halfsize, 0.0)

    return cond1 and cond2 and cond3 and cond4 and cond5 and cond6


def _find_interval(t: jax.Array, boundaries: jax.Array):
    """Find the interval of `boundaries` between which `t` lies.

    Args:
        t: Scalar float (e.g. time)
        boundaries: Array of floats

    Example: (from `test_jcalc.py`)
        >> _find_interval(1.5, jnp.array([0.0, 1.0, 2.0])) -> 2
        >> _find_interval(0.5, jnp.array([0.0])) -> 1
        >> _find_interval(-0.5, jnp.array([0.0])) -> 0
    """
    assert boundaries.ndim == 1

    @jax.vmap
    def leq_than_boundary(boundary: jax.Array):
        return jnp.where(t >= boundary, 1, 0)

    return jnp.sum(leq_than_boundary(boundaries))


def join_motionconfigs(
    configs: list[MotionConfig], boundaries: list[float]
) -> MotionConfig:
    """
    Joins multiple `MotionConfig` objects in time, transitioning between them at specified boundaries.

    This function takes a list of `MotionConfig` instances and a corresponding list of boundary times,
    and constructs a new `MotionConfig` that varies in time according to the provided segments.

    Args:
        configs (list[MotionConfig]): A list of `MotionConfig` objects to be joined.
        boundaries (list[float]): A list of time values where transitions between `configs` occur.
            Must have one element less than `configs`, as each boundary defines the transition point
            between two consecutive configurations.

    Returns:
        MotionConfig: A new `MotionConfig` object where time-dependent fields transition based on the
        specified boundaries.

    Raises:
        AssertionError: If the number of boundaries does not match `len(configs) - 1`.
        AssertionError: If time-independent fields have differing values across `configs`.

    Notes:
        - Only fields that are time-dependent (`float | TimeDependentFloat`) will change over time.
        - Time-independent fields must be the same in all `configs`, or an error is raised.
    """  # noqa: E501
    # to avoid a circular import due to `ring.utils.randomize_sys` importing `jcalc`
    from ring.utils import tree_equal

    assert len(configs) == (
        len(boundaries) + 1
    ), "length of `boundaries` should be one less than length of `configs`"
    boundaries = jnp.array(boundaries, dtype=float)

    def new_value(field: str):
        scalar_options = jnp.array([getattr(c, field) for c in configs])

        def scalar(t):
            return jax.lax.dynamic_index_in_dim(
                scalar_options, _find_interval(t, boundaries), keepdims=False
            )

        return scalar

    hints = get_type_hints(MotionConfig())
    attrs = MotionConfig().__dict__
    is_time_dependent_field = lambda key: hints[key] == (float | TimeDependentFloat)
    time_dependent_fields = [key for key in attrs if is_time_dependent_field(key)]
    time_independent_fields = [key for key in attrs if not is_time_dependent_field(key)]

    for time_dep_field in time_independent_fields:
        try:
            field_values = set([getattr(config, time_dep_field) for config in configs])
            assert (
                len(field_values) == 1
            ), f"MotionConfig.{time_dep_field}={field_values}. "
            "Should be one unique value.."
        except (
            TypeError
        ):  # dict is not hashable so test equality of all elements differently
            comparison_ele = getattr(configs[0], time_dep_field)
            for other_config in configs[1:]:
                other_ele = getattr(other_config, time_dep_field)
                assert tree_equal(
                    comparison_ele, other_ele
                ), f"MotionConfig.{time_dep_field} with {comparison_ele} != {other_ele}"
                " Should be one unique value.."

    changes = {field: new_value(field) for field in time_dependent_fields}
    return replace(configs[0], **changes)


DRAW_FN = Callable[
    # config, key_t, key_value, dt, N, params
    [
        MotionConfig,
        jax.random.PRNGKey,
        jax.random.PRNGKey,
        float | jax.Array,
        int | None,
        jax.Array,
    ],
    jax.Array,
]
P_CONTROL_TERM = Callable[
    # q, q_ref -> qdd
    # (q_size,), (q_size), -> (qd_size,)
    [jax.Array, jax.Array],
    jax.Array,
]
# this function is used to generate the velocity reference trajectory from the
# reference trajectory q, which both are required for the pd control, which it is
# required if the simulation is not kinematic but dynamic
QD_FROM_Q = Callable[
    # qs, dt -> dqs
    # (N, q_size), (1,) -> (N, qd_size)
    [jax.Array, jax.Array],
    jax.Array,
]
# used by ring.algorithms.inverse_kinematics_endeffector to  maps from
# [-inf, inf] -> feasible joint value range. Defaults to {}.
# For example: By default, for a hinge joint it uses `maths.wrap_to_pi`.
# For a spherical joint it would normalize to create a unit quaternion.
COORDINATE_VECTOR_TO_Q = Callable[
    # (q_size,) -> (q_size)
    [jax.Array],
    jax.Array,
]

# used only by `sim2real.project_xs`, and it receives a transform object
# and projects it into the feasible subspace as defined by the joint
# and returns the new transform object
PROJECT_TRANSFORM_TO_FEASIBLE = Callable[
    # base.Transform, Pytree (joint_params)
    [base.Transform, tree_utils.PyTree],
    base.Transform,
]

# used by ring.System.from_xml and by ring.RCMG
# (key) -> Pytree
# if it is not given and None, then there will be no specific
# joint_parameters for the custom joint and it will simply receive
# the defaults parameters, that is joint_params['default']
INIT_JOINT_PARAMS = Callable[[jax.Array], tree_utils.PyTree]

# (transform2_p_to_i, joint_params) -> (q_size)
INV_KIN = Callable[[base.Transform, tree_utils.PyTree], jax.Array]


@dataclass
class JointModel:
    """
    Represents the kinematic and dynamic properties of a joint type.

    A `JointModel` defines the mathematical functions required to compute joint
    transformations, motion, control terms, and inverse kinematics. It is used to
    describe the behavior of various joint types, including revolute, prismatic,
    spherical, and free joints.

    Attributes:
        transform (Callable[[jax.Array, jax.Array], base.Transform]):
            Computes the transformation (position and orientation) of the joint
            given the joint state `q` and joint parameters.

        motion (list[base.Motion | Callable[[jax.Array], base.Motion]]):
            Defines the joint motion model. It can be a list of `Motion` objects
            or callables that return `Motion` based on joint parameters.

        rcmg_draw_fn (Optional[DRAW_FN]):
            Function used to generate a reference motion trajectory for the joint
            using Randomized Control Motion Generation (RCMG).

        p_control_term (Optional[P_CONTROL_TERM]):
            Function that computes the proportional control term for the joint.

        qd_from_q (Optional[QD_FROM_Q]):
            Function to compute joint velocity (`qd`) from joint positions (`q`).

        coordinate_vector_to_q (Optional[COORDINATE_VECTOR_TO_Q]):
            Function that maps a coordinate vector to a valid joint state `q`,
            ensuring constraints (e.g., wrapping angles or normalizing quaternions).

        inv_kin (Optional[INV_KIN]):
            Function that computes the inverse kinematics for the joint, mapping
            a desired transform to joint coordinates `q`.

        init_joint_params (Optional[INIT_JOINT_PARAMS]):
            Function that initializes joint-specific parameters.

        utilities (Optional[dict[str, Any]]):
            Additional utility functions or metadata related to the joint model.

    Notes:
        - The `transform` function is essential for computing the joint's spatial
          transformation based on its generalized coordinates.
        - The `motion` attribute describes how forces and torques affect the joint.
        - The `rcmg_draw_fn` is used for RCMG motion generation.
        - The `coordinate_vector_to_q` is critical for maintaining valid joint states.
    """  # noqa: E501

    # (q, params) -> Transform
    transform: Callable[[jax.Array, jax.Array], base.Transform]
    # len(motion) == len(qd)
    # if callable: joint_params -> base.Motion
    motion: list[base.Motion | Callable[[jax.Array], base.Motion]] = field(
        default_factory=lambda: []
    )
    # (config, key_t, key_value, params) -> jax.Array
    rcmg_draw_fn: Optional[DRAW_FN] = None

    # only used by `pd_control`
    p_control_term: Optional[P_CONTROL_TERM] = None
    qd_from_q: Optional[QD_FROM_Q] = None

    # used by
    # -`inverse_kinematics_endeffector`
    # - System.coordinate_vector_to_q
    coordinate_vector_to_q: Optional[COORDINATE_VECTOR_TO_Q] = None

    # only used by `inverse_kinematics`
    inv_kin: Optional[INV_KIN] = None

    init_joint_params: Optional[INIT_JOINT_PARAMS] = None

    utilities: Optional[dict[str, Any]] = field(default_factory=lambda: dict())


def _free_transform(q, _):
    rot, pos = q[:4], q[4:]
    return base.Transform(pos, rot)


def _free_2d_transform(q, _):
    angle_x, pos_yz = q[0], q[1:]
    rot = maths.quat_rot_axis(maths.x_unit_vector, angle_x)
    pos = jnp.concatenate((jnp.array([0.0]), pos_yz))
    return base.Transform(pos, rot)


def _rxyz_transform(q, _, axis):
    q = jnp.squeeze(q)
    rot = maths.quat_rot_axis(axis, q)
    return base.Transform.create(rot=rot)


def _pxyz_transform(q, _, direction):
    pos = direction * q
    return base.Transform.create(pos=pos)


def _frozen_transform(_, __):
    return base.Transform.zero()


def _spherical_transform(q, _):
    return base.Transform.create(rot=q)


def _saddle_transform(q, _):
    rot = maths.euler_to_quat(jnp.array([0.0, q[0], q[1]]))
    return base.Transform.create(rot=rot)


def _p3d_transform(q, _):
    return base.Transform.create(pos=q)


def _cor_transform(q, _):
    free = _free_transform(q[:7], _)
    p3d = _p3d_transform(q[7:], _)
    return algebra.transform_mul(p3d, free)


mrx = base.Motion.create(ang=jnp.array([1.0, 0, 0]))
mry = base.Motion.create(ang=jnp.array([0.0, 1, 0]))
mrz = base.Motion.create(ang=jnp.array([0.0, 0, 1]))
mpx = base.Motion.create(vel=jnp.array([1.0, 0, 0]))
mpy = base.Motion.create(vel=jnp.array([0.0, 1, 0]))
mpz = base.Motion.create(vel=jnp.array([0.0, 0, 1]))


def _draw_rxyz(
    config: MotionConfig,
    key_t: jax.random.PRNGKey,
    key_value: jax.random.PRNGKey,
    dt: float | jax.Array,
    N: int | None,
    _: jax.Array,
    # TODO, delete these args and pass a modifified `config` with `replace` instead
    enable_range_of_motion: bool = True,
    free_spherical: bool = False,
    # how often it should try to fullfill the dang_min/max and delta_ang_min/max conds
    max_iter: int = 5,
) -> jax.Array:
    key_value, consume = jax.random.split(key_value)
    ANG_0 = jax.random.uniform(consume, minval=config.ang0_min, maxval=config.ang0_max)
    # `random_angle_over_time` always returns wrapped angles, thus it would be
    # inconsistent to allow an initial value that is not wrapped
    ANG_0 = maths.wrap_to_pi(ANG_0)
    return _random.random_angle_over_time(
        key_t,
        key_value,
        ANG_0,
        config.dang_min_free_spherical if free_spherical else config.dang_min,
        config.dang_max_free_spherical if free_spherical else config.dang_max,
        config.delta_ang_min_free_spherical if free_spherical else config.delta_ang_min,
        config.delta_ang_max_free_spherical if free_spherical else config.delta_ang_max,
        config.t_min,
        config.t_max,
        config.T,
        dt,
        N,
        max_iter,
        config.randomized_interpolation_angle,
        config.range_of_motion_hinge if enable_range_of_motion else False,
        config.range_of_motion_hinge_method,
        config.rom_halfsize,
        config.cdf_bins_min,
        config.cdf_bins_max,
        config.interpolation_method,
        config.include_standstills_prob,
        config.include_standstills_t_min,
        config.include_standstills_t_max,
    )


def _draw_pxyz(
    config: MotionConfig,
    key_t: jax.random.PRNGKey,
    key_value: jax.random.PRNGKey,
    dt: float | jax.Array,
    N: int | None,
    __: jax.Array,
    cor: bool = False,
) -> jax.Array:
    key_value, consume = jax.random.split(key_value)
    POS_0 = jax.random.uniform(
        consume,
        minval=config.cor_pos0_min if cor else config.pos0_min,
        maxval=config.cor_pos0_max if cor else config.pos0_max,
    )
    max_iter = 100
    return _random.random_position_over_time(
        key_t,
        key_value,
        POS_0,
        config.cor_pos_min if cor else config.pos_min,
        config.cor_pos_max if cor else config.pos_max,
        config.cor_dpos_min if cor else config.dpos_min,
        config.cor_dpos_max if cor else config.dpos_max,
        config.cor_t_min if cor else config.t_min,
        config.cor_t_max if cor else config.t_max,
        config.T,
        dt,
        N,
        max_iter,
        config.randomized_interpolation_position,
        config.cdf_bins_min,
        config.cdf_bins_max,
        config.interpolation_method,
        config.include_standstills_prob,
        config.include_standstills_t_min,
        config.include_standstills_t_max,
    )


def _draw_spherical(
    config: MotionConfig,
    key_t: jax.random.PRNGKey,
    key_value: jax.random.PRNGKey,
    dt: float | jax.Array,
    N: int | None,
    _: jax.Array,
) -> jax.Array:
    # NOTE: We draw 3 euler angles and then build a quaternion.
    # Not ideal, but i am unaware of a better way.
    def draw_euler_angles(key_t, key_value):
        return _draw_rxyz(
            config,
            key_t,
            key_value,
            dt,
            N,
            None,
            enable_range_of_motion=False,
            free_spherical=True,
        )

    euler_angles = jax.vmap(draw_euler_angles, in_axes=(None, 0))(
        key_t, jax.random.split(key_value, 3)
    ).T
    q = maths.quat_euler(euler_angles)
    return q


def _draw_saddle(
    config: MotionConfig,
    key_t: jax.random.PRNGKey,
    key_value: jax.random.PRNGKey,
    dt: float | jax.Array,
    N: int | None,
    _: jax.Array,
) -> jax.Array:
    def draw_euler_angles(key_t, key_value):
        return _draw_rxyz(
            config,
            key_t,
            key_value,
            dt,
            N,
            None,
            enable_range_of_motion=False,
            free_spherical=False,
        )

    yz_euler_angles = jax.vmap(draw_euler_angles, in_axes=(None, 0))(
        key_t, jax.random.split(key_value)
    ).T
    return yz_euler_angles


def _draw_p3d_and_cor(
    config: MotionConfig,
    key_t: jax.random.PRNGKey,
    key_value: jax.random.PRNGKey,
    dt: float | jax.Array,
    N: int | None,
    __: jax.Array,
    cor: bool,
) -> jax.Array:
    keys = jax.random.split(key_value, 3)

    def draw(key_value, xyz: str):
        return _draw_pxyz(
            replace(
                config,
                pos_min=getattr(config, f"pos_min_p3d_{xyz}"),
                pos_max=getattr(config, f"pos_max_p3d_{xyz}"),
                pos0_min=getattr(config, f"pos0_min_p3d_{xyz}"),
                pos0_max=getattr(config, f"pos0_max_p3d_{xyz}"),
            ),
            key_t,
            key_value,
            dt,
            N,
            None,
            cor,
        )[:, None]

    px, py, pz = draw(keys[0], "x"), draw(keys[1], "y"), draw(keys[2], "z")
    return jnp.concat((px, py, pz), axis=-1)


def _draw_p3d(
    config: MotionConfig,
    key_t: jax.random.PRNGKey,
    key_value: jax.random.PRNGKey,
    dt: float | jax.Array,
    N: int | None,
    __: jax.Array,
) -> jax.Array:
    return _draw_p3d_and_cor(config, key_t, key_value, dt, N, None, cor=False)


def _draw_cor(
    config: MotionConfig,
    key_t: jax.random.PRNGKey,
    key_value: jax.random.PRNGKey,
    dt: float | jax.Array,
    N: int | None,
    __: jax.Array,
) -> jax.Array:
    key_value1, key_value2 = jax.random.split(key_value)
    q_free = _draw_free(config, key_t, key_value1, dt, N, None)
    q_p3d = _draw_p3d_and_cor(config, key_t, key_value2, dt, N, None, cor=True)
    return jnp.concatenate((q_free, q_p3d), axis=1)


def _draw_free(
    config: MotionConfig,
    key_t: jax.random.PRNGKey,
    key_value: jax.random.PRNGKey,
    dt: float | jax.Array,
    N: int | None,
    __: jax.Array,
) -> jax.Array:
    key_value1, key_value2 = jax.random.split(key_value)
    q = _draw_spherical(config, key_t, key_value1, dt, N, None)
    pos = _draw_p3d(config, key_t, key_value2, dt, N, None)
    return jnp.concatenate((q, pos), axis=1)


def _draw_free_2d(
    config: MotionConfig,
    key_t: jax.random.PRNGKey,
    key_value: jax.random.PRNGKey,
    dt: float | jax.Array,
    N: int | None,
    __: jax.Array,
) -> jax.Array:
    key_value1, key_value2 = jax.random.split(key_value)
    angle_x = _draw_rxyz(
        config,
        key_t,
        key_value1,
        dt,
        N,
        None,
        enable_range_of_motion=False,
        free_spherical=True,
    )[:, None]
    pos_yz = _draw_p3d(config, key_t, key_value2, dt, N, None)[:, :2]
    return jnp.concatenate((angle_x, pos_yz), axis=1)


def _draw_frozen(
    config: MotionConfig, _, __, dt: float | jax.Array, N: int | None, ___
) -> jax.Array:
    if N is None:
        N = int(config.T / dt)
    return jnp.zeros((N, 0))


qrel = lambda q1, q2: maths.quat_mul(q1, maths.quat_inv(q2))


def _qd_from_q_quaternion(qs, dt):
    axis, angle = maths.quat_to_rot_axis(qrel(qs[2:], qs[:-2]))
    # axis.shape = (n_timesteps, 3); angle.shape = (n_timesteps,)
    # Thus add singleton dimesions otherwise broadcast error
    dq = axis * angle[:, None] / (2 * dt)
    dq = jnp.vstack((jnp.zeros((3,)), dq, jnp.zeros((3,))))
    return dq


def _qd_from_q_cartesian(qs, dt):
    dq = jnp.vstack(
        (jnp.zeros_like(qs[0]), (qs[2:] - qs[:-2]) / (2 * dt), jnp.zeros_like(qs[0]))
    )
    return dq


def _p_control_quaternion(q, q_ref):
    axis, angle = maths.quat_to_rot_axis(qrel(q_ref, q))
    return axis * angle


def _p_control_term_rxyz(q, q_ref):
    # q_ref comes from rcmg. Thus, it is already wrapped
    # TODO: Currently state.q is not wrapped. Change that?
    return maths.wrap_to_pi(q_ref - maths.wrap_to_pi(q))


def _p_control_term_pxyz_p3d(q, q_ref):
    return q_ref - q


def _p_control_term_frozen(q, q_ref):
    return jnp.array([])


def _p_control_term_spherical(q, q_ref):
    return _p_control_quaternion(q, q_ref)


def _p_control_term_free(q, q_ref):
    return jnp.concatenate(
        (
            _p_control_quaternion(q[:4], q_ref[:4]),
            (q_ref[4:] - q[4:]),
        )
    )


def _p_control_term_free_2d(q, q_ref):
    return jnp.concatenate(
        (
            _p_control_term_rxyz(q[:1], q_ref[:1]),
            (q_ref[1:] - q[1:]),
        )
    )


def _p_control_term_cor(q, q_ref):
    return _p_control_term_free(q, q_ref)


def _qd_from_q_free(qs, dt):
    qd_quat = _qd_from_q_quaternion(qs[:, :4], dt)
    qd_pos = _qd_from_q_cartesian(qs[:, 4:], dt)
    return jnp.hstack((qd_quat, qd_pos))


def _coordinate_vector_to_q_free_spherical_cor(q):
    return q.at[:4].set(maths.safe_normalize(q[:4]))


def _coordinate_vector_to_q_free_2d(q):
    return q.at[0].set(maths.wrap_to_pi(q[0]))


_str2idx = {"x": slice(0, 1), "y": slice(1, 2), "z": slice(2, 3)}


def _inv_kin_rxyz_factory(xyz: str):
    k = maths.unit_vectors(xyz)

    def _inv_kin_rxyz(x: base.Transform, _) -> jax.Array:
        # TODO
        # NOTE: CONVENTION
        # the first return is the much faster version but it suffers from a convention
        # issue the second version is equivalent and does not suffer from the
        # convention issue but it is much slower
        q = x.rot
        angle = 2 * jnp.arctan2(q[1:] @ k, q[0])
        return -angle[None]
        axis, angle = maths.quat_to_rot_axis(maths.quat_project(q, k)[0])
        return jnp.where((k @ axis) > 0, angle, -angle)[None]

    return _inv_kin_rxyz


def _inv_kin_pxyz_factory(xyz: str):
    idx = _str2idx[xyz]

    def _inv_kin_pxyz(x: base.Transform, _) -> jax.Array:
        return x.pos[idx]

    return _inv_kin_pxyz


def _inv_kin_free_2d(x: base.Transform, _) -> jax.Array:
    angle_x = _inv_kin_rxyz_factory("x")
    return jnp.concatenate((angle_x(x), x.pos[1:]))


_joint_types = {
    "free": JointModel(
        _free_transform,
        [mrx, mry, mrz, mpx, mpy, mpz],
        _draw_free,
        _p_control_term_free,
        _qd_from_q_free,
        coordinate_vector_to_q=_coordinate_vector_to_q_free_spherical_cor,
        inv_kin=lambda x, _: jnp.concatenate((x.rot, x.pos)),
    ),
    "free_2d": JointModel(
        _free_2d_transform,
        [mrx, mpy, mpz],
        _draw_free_2d,
        _p_control_term_free_2d,
        _qd_from_q_cartesian,
        coordinate_vector_to_q=_coordinate_vector_to_q_free_2d,
        inv_kin=_inv_kin_free_2d,
    ),
    "frozen": JointModel(
        _frozen_transform,
        [],
        _draw_frozen,
        _p_control_term_frozen,
        _qd_from_q_cartesian,
        lambda q: q,
        lambda x, _: jnp.array([]),
    ),
    "spherical": JointModel(
        _spherical_transform,
        [mrx, mry, mrz],
        _draw_spherical,
        _p_control_term_spherical,
        _qd_from_q_quaternion,
        _coordinate_vector_to_q_free_spherical_cor,
        lambda x, _: x.rot,
    ),
    "p3d": JointModel(
        _p3d_transform,
        [mpx, mpy, mpz],
        _draw_p3d,
        _p_control_term_pxyz_p3d,
        _qd_from_q_cartesian,
        lambda q: q,
        lambda x, _: x.pos,
    ),
    "cor": JointModel(
        _cor_transform,
        [mrx, mry, mrz, mpx, mpy, mpz, mpx, mpy, mpz],
        _draw_cor,
        _p_control_term_cor,
        _qd_from_q_free,
        _coordinate_vector_to_q_free_spherical_cor,
    ),
    "rx": JointModel(
        lambda q, _: _rxyz_transform(q, _, jnp.array([1.0, 0, 0])),
        [mrx],
        _draw_rxyz,
        _p_control_term_rxyz,
        _qd_from_q_cartesian,
        maths.wrap_to_pi,
        _inv_kin_rxyz_factory("x"),
    ),
    "ry": JointModel(
        lambda q, _: _rxyz_transform(q, _, jnp.array([0.0, 1, 0])),
        [mry],
        _draw_rxyz,
        _p_control_term_rxyz,
        _qd_from_q_cartesian,
        maths.wrap_to_pi,
        _inv_kin_rxyz_factory("y"),
    ),
    "rz": JointModel(
        lambda q, _: _rxyz_transform(q, _, jnp.array([0.0, 0, 1])),
        [mrz],
        _draw_rxyz,
        _p_control_term_rxyz,
        _qd_from_q_cartesian,
        maths.wrap_to_pi,
        _inv_kin_rxyz_factory("z"),
    ),
    "px": JointModel(
        lambda q, _: _pxyz_transform(q, _, jnp.array([1.0, 0, 0])),
        [mpx],
        _draw_pxyz,
        _p_control_term_pxyz_p3d,
        _qd_from_q_cartesian,
        lambda q: q,
        _inv_kin_pxyz_factory("x"),
    ),
    "py": JointModel(
        lambda q, _: _pxyz_transform(q, _, jnp.array([0.0, 1, 0])),
        [mpy],
        _draw_pxyz,
        _p_control_term_pxyz_p3d,
        _qd_from_q_cartesian,
        lambda q: q,
        _inv_kin_pxyz_factory("y"),
    ),
    "pz": JointModel(
        lambda q, _: _pxyz_transform(q, _, jnp.array([0.0, 0, 1])),
        [mpz],
        _draw_pxyz,
        _p_control_term_pxyz_p3d,
        _qd_from_q_cartesian,
        lambda q: q,
        _inv_kin_pxyz_factory("z"),
    ),
    "saddle": JointModel(
        _saddle_transform,
        [mry, mrz],
        _draw_saddle,
        _p_control_term_rxyz,
        _qd_from_q_cartesian,
        maths.wrap_to_pi,
    ),
}


def get_joint_model(joint_type: str) -> JointModel:
    assert (
        joint_type in _joint_types
    ), f"{joint_type} not in {list(_joint_types.keys())}"
    return _joint_types[joint_type]


def register_new_joint_type(
    joint_type: str,
    joint_model: JointModel,
    q_width: int,
    qd_width: Optional[int] = None,
    overwrite: bool = False,
):
    """
    Registers a new joint type with its corresponding `JointModel` and kinematic properties.

    This function allows the addition of custom joint types to the system by associating
    them with a `JointModel`, specifying their state and velocity dimensions, and optionally
    overwriting existing joint definitions.

    Args:
        joint_type (str):
            Name of the new joint type to register.
        joint_model (JointModel):
            The `JointModel` instance defining the kinematic and dynamic properties of the joint.
        q_width (int):
            Number of generalized coordinates (degrees of freedom) required to represent the joint.
        qd_width (Optional[int], default=None):
            Number of velocity coordinates associated with the joint. Defaults to `q_width`.
        overwrite (bool, default=False):
            If `True`, allows overwriting an existing joint type. Otherwise, raises an error if
            the joint type already exists.

    Raises:
        AssertionError:
            - If `joint_type` is `"default"` (reserved name).
            - If `joint_type` already exists and `overwrite=False`.
            - If `qd_width` is not provided and does not default to `q_width`.
            - If `joint_model.motion` length does not match `qd_width`.

    Notes:
        - The function updates global dictionaries that store joint properties, including:
          - `_joint_types`: Maps joint type names to `JointModel` instances.
          - `base.Q_WIDTHS`: Stores the number of state coordinates for each joint type.
          - `base.QD_WIDTHS`: Stores the number of velocity coordinates for each joint type.
        - If `overwrite=True`, existing entries are removed before adding the new joint type.
        - Ensures consistency between motion definitions and velocity coordinate dimensions.

    Example:
        ```python
        new_joint = JointModel(
            transform=my_transform_fn,
            motion=[base.Motion.create(ang=jnp.array([1, 0, 0]))],
        )
        register_new_joint_type("custom_hinge", new_joint, q_width=1)
        ```
    """  # noqa: E501
    # this name is used
    assert joint_type != "default", "Please use another name."

    exists = joint_type in _joint_types
    if exists and overwrite:
        for dic in [
            base.Q_WIDTHS,
            base.QD_WIDTHS,
            _joint_types,
        ]:
            dic.pop(joint_type)
    else:
        assert (
            not exists
        ), f"joint type `{joint_type}`already exists, use `overwrite=True`"

    if qd_width is None:
        qd_width = q_width

    assert len(joint_model.motion) == qd_width

    _joint_types.update({joint_type: joint_model})
    base.Q_WIDTHS.update({joint_type: q_width})
    base.QD_WIDTHS.update({joint_type: qd_width})


def _limit_scope_of_joint_params(
    joint_type: str, joint_params: dict[str, tree_utils.PyTree]
) -> tree_utils.PyTree:
    if joint_type not in joint_params:
        return joint_params["default"]
    else:
        return joint_params[joint_type]


def jcalc_transform(
    joint_type: str, q: jax.Array, joint_params: dict[str, tree_utils.PyTree]
) -> base.Transform:
    joint_params = _limit_scope_of_joint_params(joint_type, joint_params)
    return _joint_types[joint_type].transform(q, joint_params)


def _to_motion(
    m: base.Motion | Callable[[jax.Array], base.Motion], joint_params: tree_utils.PyTree
) -> base.Motion:
    if isinstance(m, base.Motion):
        return m
    return m(joint_params)


def jcalc_motion(
    joint_type: str, qd: jax.Array, joint_params: dict[str, tree_utils.PyTree]
) -> base.Motion:
    joint_params = _limit_scope_of_joint_params(joint_type, joint_params)
    list_motion = _joint_types[joint_type].motion
    m = base.Motion.zero()
    for dof in range(len(list_motion)):
        m += _to_motion(list_motion[dof], joint_params) * qd[dof]
    return m


def jcalc_tau(
    joint_type: str, f: base.Force, joint_params: dict[str, tree_utils.PyTree]
) -> jax.Array:
    joint_params = _limit_scope_of_joint_params(joint_type, joint_params)
    list_motion = _joint_types[joint_type].motion
    return jnp.array(
        [algebra.motion_dot(_to_motion(m, joint_params), f) for m in list_motion]
    )


def _init_joint_params(key: jax.Array, sys: base.System) -> base.System:
    """Search systems for custom joints and call their JointModel.init_joint_params
    functions. Then return updated system."""

    joint_params_init_fns = {}
    for typ in sys.link_types:
        if typ not in joint_params_init_fns:
            init_joint_params = _joint_types[typ].init_joint_params
            if init_joint_params is not None:
                joint_params_init_fns[typ] = init_joint_params

    joint_params: dict[str, tree_utils.PyTree] = {}
    n_links = sys.num_links()
    for typ in joint_params_init_fns:
        keys = jax.random.split(key, num=n_links + 1)
        key, consume = keys[0], keys[1:]
        joint_params[typ] = jax.vmap(joint_params_init_fns[typ])(consume)

    # add batch default parameters
    joint_params["default"] = jnp.zeros((n_links, 0))

    return sys.replace(links=sys.links.replace(joint_params=joint_params))
