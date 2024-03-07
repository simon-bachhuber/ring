from dataclasses import dataclass

import jax
import jax.numpy as jnp
import ring
from ring import maths
from ring.algorithms._random import random_angle_over_time


def deg2rad(deg: float):
    return (deg / 180.0) * 3.1415926535


_DEFAULT_LENGTH_RAD = deg2rad(80.0)


@dataclass
class SuntayConfig:
    flexion_rot_min: float = -deg2rad(5.0)
    flexion_rot_max: float = deg2rad(95.0)
    flexion_pos_length_rad: float = _DEFAULT_LENGTH_RAD
    flexion_pos_min: float = -0.015
    flexion_pos_max: float = 0.015
    flexion_restrict_method: str = "minmax"
    abduction_rot_length_rad: float = _DEFAULT_LENGTH_RAD
    abduction_rot_min: float = deg2rad(-4)
    abduction_rot_max: float = deg2rad(4)
    abduction_pos_length_rad: float = _DEFAULT_LENGTH_RAD
    abduction_pos_min: float = -0.015
    abduction_pos_max: float = 0.015
    external_rot_length_rad: float = _DEFAULT_LENGTH_RAD
    external_rot_min: float = deg2rad(-10)
    external_rot_max: float = deg2rad(10)
    external_pos_length_rad: float = _DEFAULT_LENGTH_RAD
    external_pos_min: float = -0.06
    external_pos_max: float = 0.0
    num_points_gps: int = 50
    large_abs_values_of_gps: float = 1 / 4


def register_suntay(sconfig: SuntayConfig, name: str = "suntay"):
    """Ref to 'E.S. Grood and W.J. Suntay' paper"""

    flexion_xs = jnp.linspace(
        sconfig.flexion_rot_min, sconfig.flexion_rot_max, num=sconfig.num_points_gps
    )

    def _q_nonflexion(q_flexion, params):
        nonflexion_ys = params
        nonflexion_q = jnp.interp(q_flexion, flexion_xs, nonflexion_ys)
        return nonflexion_q

    def _suntay_rotation_matrix_R_transpose_eq26(alpha, beta, gamma):
        sin_alp, sin_bet, sin_gam = jnp.sin(alpha), jnp.sin(beta), jnp.sin(gamma)
        cos_alp, cos_bet, cos_gam = jnp.cos(alpha), jnp.cos(beta), jnp.cos(gamma)
        return jnp.array(
            [
                [cos_gam * sin_bet, sin_gam * sin_bet, cos_bet],
                [
                    -cos_alp * sin_gam - cos_gam * sin_alp * cos_bet,
                    cos_alp * cos_gam - sin_gam * sin_alp * cos_bet,
                    sin_bet * sin_alp,
                ],
                [
                    sin_alp * sin_gam - cos_gam * cos_alp * cos_bet,
                    -cos_gam * sin_alp - cos_alp * cos_bet * sin_gam,
                    cos_alp * sin_bet,
                ],
            ]
        ).T

    def _suntay_translation_vector_H_eq9(alpha, beta, S):
        sin_alp, sin_bet = jnp.sin(alpha), jnp.sin(beta)
        cos_alp, cos_bet = jnp.cos(alpha), jnp.cos(beta)
        # eq (10)
        U = jnp.array(
            [
                [1, 0, cos_bet],
                [0, cos_alp, sin_alp * sin_bet],
                [0, -sin_alp, cos_alp * sin_bet],
            ]
        )
        return U @ S

    def _alpha_beta_gamma_S(q_flexion, params):
        assert q_flexion.shape == (1,)

        # (1,) -> (,)
        q_flexion = q_flexion[0]

        S = jnp.stack(
            [_q_nonflexion(q_flexion, params[f"ys_S{i}"]) for i in range(1, 4)]
        )
        # table 2 of suntay paper
        alpha = q_flexion
        # note the minus sign, because in config we specify `abduction` not `adduction`
        adduction = -_q_nonflexion(q_flexion, params["ys_beta"])
        beta = jnp.pi / 2 + adduction
        gamma = _q_nonflexion(q_flexion, params["ys_gamma"])
        return alpha, beta, gamma, S

    def _utils_find_suntay_joint(sys: ring.System) -> str:
        suntay_link_name = None
        for link_name, link_type in zip(sys.link_names, sys.link_types):
            if link_type == name:
                if suntay_link_name is not None:
                    raise Exception(
                        f"multiple links of type `{name}` found, link_names "
                        f"are [{suntay_link_name}, {link_name}]"
                    )
                suntay_link_name = link_name

        if suntay_link_name is None:
            raise Exception(
                f"no link with type `{name}` found, link_types are {sys.link_types}"
            )
        return suntay_link_name

    def _utils_Q_S_H_alpha_beta_gamma(sys: ring.System, qs: jax.Array):
        # qs.shape = (timesteps, q_size)
        assert qs.ndim == 2
        assert qs.shape[-1] == sys.q_size()

        suntay_link_name = _utils_find_suntay_joint(sys)

        params = jax.tree_map(
            lambda arr: arr[sys.idx_map("l")[suntay_link_name]],
            sys.links.joint_params[name],
        )
        # shape = (timesteps, 1)
        q_flexion = qs[:, sys.idx_map("q")[suntay_link_name]]

        @jax.vmap
        def _Q_S_H_alpha_beta_gamma_from_q_flexion(q_flexion):
            alpha, beta, gamma, S = _alpha_beta_gamma_S(q_flexion, params)
            cos_bet = jnp.cos(beta)
            Q = jnp.array([S[0] + S[2] * cos_bet, S[1], -S[2] - S[0] * cos_bet])
            # translation from femur to tibia
            H = _suntay_translation_vector_H_eq9(alpha, beta, S)
            return Q, S, H, alpha, beta, gamma

        return _Q_S_H_alpha_beta_gamma_from_q_flexion(q_flexion)

    def _transform_suntay(q_flexion, params):
        alpha, beta, gamma, S = _alpha_beta_gamma_S(q_flexion, params)

        # rotation from femur to tibia
        R_T = _suntay_rotation_matrix_R_transpose_eq26(alpha, beta, gamma)
        q_fem_tib = maths.quat_from_3x3(R_T)
        # translation from femur to tibia
        H = _suntay_translation_vector_H_eq9(alpha, beta, S)

        return ring.Transform.create(pos=H, rot=q_fem_tib)

    def _draw_and_rom(key, ys, length, mn, mx):
        randomized_ys = gp_draw(key, flexion_xs, ys, length)
        amin, amax = -sconfig.large_abs_values_of_gps, sconfig.large_abs_values_of_gps
        if ys is not None:
            amin += jnp.min(ys)
            amax += jnp.max(ys)
        return restrict(randomized_ys, mn, mx, amin, amax)

    def _init_joint_params_suntay(key):
        params = dict()
        for config_name, params_name in zip(
            [
                "flexion_pos",
                "abduction_rot",
                "abduction_pos",
                "external_rot",
                "external_pos",
            ],
            ["ys_S1", "ys_beta", "ys_S2", "ys_gamma", "ys_S3"],
        ):
            key, consume = jax.random.split(key)
            # TODO, ys=None!
            get = lambda cnfkey: getattr(sconfig, config_name + cnfkey)
            params[params_name] = _draw_and_rom(
                consume, None, get("_length_rad"), get("_min"), get("_max")
            )

        return params

    def _draw_flexion_angle(
        mconfig: ring.MotionConfig,
        key_t: jax.random.PRNGKey,
        key_value: jax.random.PRNGKey,
        dt: float,
        _: jax.Array,
    ) -> jax.Array:
        key_value, consume = jax.random.split(key_value)
        ANG_0 = jax.random.uniform(
            consume, minval=mconfig.ang0_min, maxval=mconfig.ang0_max
        )
        # `random_angle_over_time` always returns wrapped angles, thus it would be
        # inconsistent to allow an initial value that is not wrapped
        ANG_0 = maths.wrap_to_pi(ANG_0)
        qs_flexion = random_angle_over_time(
            key_t,
            key_value,
            ANG_0,
            mconfig.dang_min,
            mconfig.dang_max,
            mconfig.delta_ang_min,
            mconfig.delta_ang_max,
            mconfig.t_min,
            mconfig.t_max,
            mconfig.T,
            dt,
            5,
            mconfig.randomized_interpolation_angle,
            mconfig.range_of_motion_hinge,
            mconfig.range_of_motion_hinge_method,
            mconfig.cdf_bins_min,
            mconfig.cdf_bins_max,
            mconfig.interpolation_method,
        )
        return restrict(
            qs_flexion,
            sconfig.flexion_rot_min,
            sconfig.flexion_rot_max,
            -jnp.pi,
            jnp.pi,
            method=sconfig.flexion_restrict_method,
        )

    joint_model = ring.JointModel(
        transform=_transform_suntay,
        rcmg_draw_fn=_draw_flexion_angle,
        init_joint_params=_init_joint_params_suntay,
        utilities=dict(
            Q_S_H_alpha_beta_gamma=_utils_Q_S_H_alpha_beta_gamma,
            find_suntay_joint=_utils_find_suntay_joint,
        ),
    )
    ring.register_new_joint_type(name, joint_model, 1, qd_width=0, overwrite=True)


def gp_draw(key, xs, ys=None, length: float = 1.0, noise=0.0, method="svd", **kwargs):
    if ys is None:
        ys = jnp.zeros_like(xs)
    cov = _gp_K(lambda *args: _rbf_kernel(*args, length=length), xs, noise)
    return jax.random.multivariate_normal(
        key=key, mean=ys, cov=cov, method=method, **kwargs
    )


def _gp_K(kernel, xs, noise: float):
    assert xs.ndim == 1
    N = len(xs)
    xs = xs[:, None]

    K = jax.vmap(lambda x1: jax.vmap(lambda x2: kernel(x1, x2))(xs))(xs)
    assert K.shape == (N, N, 1)
    return K[..., 0] + jnp.eye(N) * noise


def _rbf_kernel(x1: float, x2: float, length: float):
    return jnp.exp(-((x1 - x2) ** 2) / (2 * length**2))


def _shift(ys, min, max):
    return (ys - min) / (max - min)


def _shift_inv(ys, min, max):
    return (ys * (max - min)) + min


def _normalize(ys, amin=None, amax=None):
    if amin is None:
        amin = jnp.min(ys)
    else:
        amin = jnp.min(jnp.array([amin, jnp.min(ys)]))
    if amax is None:
        amax = jnp.max(ys)
    else:
        amax = jnp.max(jnp.array([amax, jnp.max(ys)]))
    return _shift(ys, amin, amax)


def _smoothclamp(x, mi, mx):
    return mi + (mx - mi) * (
        lambda t: jnp.where(t < 0, 0, jnp.where(t <= 1, 3 * t**2 - 2 * t**3, 1))
    )((x - mi) / (mx - mi))


def _sigmoidclamp(x, mi, mx):
    return mi + (mx - mi) * (lambda t: (1 + 200 ** (-t + 0.5)) ** (-1))(
        (x - mi) / (mx - mi)
    )


def restrict(
    ys,
    min: float,
    max: float,
    actual_min=None,
    actual_max=None,
    method: str = "minmax",
    method_kwargs=dict(),
):
    if method == "minmax":
        # scale to [0, 1]
        ys = _normalize(ys, actual_min, actual_max)
        # scale to [min, max]
        return _shift_inv(ys, min, max)
    elif method == "clip":
        return jnp.clip(ys, min, max)
    elif method == "smoothclamp":
        return _smoothclamp(ys, min, max)
    elif method == "sigmoidclamp":
        return _sigmoidclamp(ys, min, max)
    elif method == "sigmoid":
        # scale to [0, 1]
        ys = _normalize(ys, actual_min, actual_max)
        # scale to [-stepness, stepness]
        stepness = method_kwargs.get("stepness", 3.0)
        ys = _shift_inv(ys, -stepness, stepness)
        # scale to [0, 1]
        ys = jax.nn.sigmoid(ys)
        return _shift_inv(ys, min, max)
    else:
        raise NotImplementedError()
