from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

import haiku as hk
import jax
import jax.numpy as jnp
from tree_utils import PyTree

import ring
from ring import maths
from ring.algorithms import jcalc
from ring.algorithms._random import random_angle_over_time

Params = PyTree


class DrawnFnPair(NamedTuple):
    # (key) -> tree
    init: Callable[[jax.Array], Params]
    # (params, q) -> (1,)
    apply: Callable[[Params, jax.Array], jax.Array]


# (flexions, min, max) -> DrawnFnPair
DrawnFnPairFactory = Callable[[jax.Array, float, float], DrawnFnPair]


def deg2rad(deg: float):
    return (deg / 180.0) * 3.1415926535


def GP_DrawFnPair(
    length_scale: float = 1.4, large_abs_values_of_gps: float = 0.25
) -> DrawnFnPairFactory:

    def factory(xs, mn, mx):
        def init(key):
            return {
                "xs": xs,
                "ys": _gp_draw_and_rom(
                    key=key,
                    xs=xs,
                    ys=None,
                    length_scale=length_scale,
                    mn=mn,
                    mx=mx,
                    amin=-large_abs_values_of_gps,
                    amax=large_abs_values_of_gps,
                ),
            }

        def apply(params, q):
            return jnp.interp(q, params["xs"], params["ys"])

        return DrawnFnPair(init, apply)

    return factory


@dataclass
class SuntayConfig:
    flexion_rot_min: float = -deg2rad(5.0)
    flexion_rot_max: float = deg2rad(95.0)
    flexion_rot_restrict_method: str = "minmax"
    ###
    flexion_pos_min: float = -0.015
    flexion_pos_max: float = 0.015
    flexion_pos_factory: DrawnFnPairFactory = GP_DrawFnPair()
    ###
    abduction_rot_min: float = deg2rad(-4)
    abduction_rot_max: float = deg2rad(4)
    abduction_rot_factory: DrawnFnPairFactory = GP_DrawFnPair()
    ###
    abduction_pos_min: float = -0.015
    abduction_pos_max: float = 0.015
    abduction_pos_factory: DrawnFnPairFactory = GP_DrawFnPair()
    ###
    external_rot_min: float = deg2rad(-10)
    external_rot_max: float = deg2rad(10)
    external_rot_factory: DrawnFnPairFactory = GP_DrawFnPair()
    ###
    external_pos_min: float = -0.06
    external_pos_max: float = 0.0
    external_pos_factory: DrawnFnPairFactory = GP_DrawFnPair()
    ###
    num_points: int = 50
    mconfig: Optional[ring.MotionConfig] = None


def register_suntay(sconfig: SuntayConfig, name: str = "suntay"):
    """Ref to 'E.S. Grood and W.J. Suntay' paper"""

    flexion_xs = jnp.linspace(
        sconfig.flexion_rot_min, sconfig.flexion_rot_max, num=sconfig.num_points
    )

    draw_fn_pairs = {}
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
        get = lambda key: getattr(sconfig, config_name + "_" + key)
        factory = get("factory")
        draw_fn_pairs[params_name] = factory(flexion_xs, get("min"), get("max"))

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

        S_123 = []
        for i in range(1, 4):
            key = f"ys_S{i}"
            S_123.append(draw_fn_pairs[key].apply(params[key], q_flexion))
        S = jnp.stack(S_123)
        # table 2 of suntay paper
        alpha = q_flexion
        # note the minus sign, because in config we specify `abduction` not `adduction`
        adduction = -draw_fn_pairs["ys_beta"].apply(params["ys_beta"], q_flexion)
        beta = jnp.pi / 2 + adduction
        gamma = draw_fn_pairs["ys_gamma"].apply(params["ys_gamma"], q_flexion)
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

    def _init_joint_params_suntay(key):
        params = dict()
        for params_name, draw_fn_pair in draw_fn_pairs.items():
            key, consume = jax.random.split(key)
            params[params_name] = draw_fn_pair.init(consume)

        return params

    def _draw_flexion_angle(
        mconfig: ring.MotionConfig,
        key_t: jax.random.PRNGKey,
        key_value: jax.random.PRNGKey,
        dt: float | jax.Array,
        N: int | None,
        _: jax.Array,
    ) -> jax.Array:
        key_value, consume = jax.random.split(key_value)

        if sconfig.mconfig is not None:
            mconfig = sconfig.mconfig

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
            N,
            5,
            mconfig.randomized_interpolation_angle,
            mconfig.range_of_motion_hinge,
            mconfig.range_of_motion_hinge_method,
            mconfig.cdf_bins_min,
            mconfig.cdf_bins_max,
            mconfig.interpolation_method,
        )
        return _apply_rom(qs_flexion)

    def _apply_rom(qs_flexion):
        return restrict(
            qs_flexion,
            sconfig.flexion_rot_min,
            sconfig.flexion_rot_max,
            -jnp.pi,
            jnp.pi,
            method=sconfig.flexion_rot_restrict_method,
        )

    def coordinate_vector_to_q_suntay(q_flexion):
        q_flexion = ring.maths.wrap_to_pi(q_flexion)
        return _apply_rom(q_flexion)

    joint_model = ring.JointModel(
        transform=_transform_suntay,
        motion=[jcalc.mrx],
        rcmg_draw_fn=_draw_flexion_angle,
        p_control_term=jcalc._p_control_term_rxyz,
        qd_from_q=jcalc._qd_from_q_cartesian,
        coordinate_vector_to_q=coordinate_vector_to_q_suntay,
        inv_kin=None,
        init_joint_params=_init_joint_params_suntay,
        utilities=dict(
            Q_S_H_alpha_beta_gamma=_utils_Q_S_H_alpha_beta_gamma,
            find_suntay_joint=_utils_find_suntay_joint,
            sconfig=sconfig,
        ),
    )
    ring.register_new_joint_type(name, joint_model, 1, overwrite=True)


def _scale_delta(method: str, key, xs, mn, mx, amin, amax, **kwargs):
    if method == "normal":
        delta = jnp.clip(jax.random.normal(key) + 0.5, 1.0)
    elif method == "uniform":
        delta = 1 / (jax.random.uniform(key) + 1e-2)
    else:
        raise NotImplementedError

    return delta


def Polynomial_DrawnFnPair(
    order: int = 2,
    center: bool = False,
    flexion_center_deg: Optional[float] = None,
    include_bias: bool = True,
    enable_scale_delta: bool = True,
    scale_delta_method: str = "normal",
    scale_delta_kwargs: dict = dict(),
) -> DrawnFnPairFactory:
    assert not (order == 0 and not include_bias)

    flexion_center = (
        jnp.deg2rad(flexion_center_deg) if flexion_center_deg is not None else None
    )
    del flexion_center_deg

    # because 0-th order is also counted
    order += 1
    powers = jnp.arange(order) if include_bias else jnp.arange(1, order)

    def factory(xs, mn, mx):
        nonlocal flexion_center

        flexion_mn = jnp.min(xs)
        flexion_mx = jnp.max(xs)

        def _apply_poly_factors(poly_factors, q):
            return poly_factors @ jnp.power(q, powers)

        if flexion_center is None:
            flexion_center = (flexion_mn + flexion_mx) / 2

        if (order - 1) == 0:
            method = "clip"
            minval, maxval = mn, mx
        else:
            method = "minmax"
            minval, maxval = -1.0, 1.0

        def init(key):
            c1, c2, c3 = jax.random.split(key, 3)
            poly_factors = jax.random.uniform(
                c1, shape=(len(powers),), minval=minval, maxval=maxval
            )
            q0 = jax.random.uniform(c2, minval=flexion_mn, maxval=flexion_mx)
            values = jax.vmap(_apply_poly_factors, in_axes=(None, 0))(
                poly_factors, xs - q0
            )
            eps = 1e-6
            amin, amax = jnp.min(values), jnp.max(values) + eps
            if enable_scale_delta:
                delta = amax - amin
                scale_delta = _scale_delta(
                    scale_delta_method, c3, xs, mn, mx, amin, amax, **scale_delta_kwargs
                )
                amax = amin + delta * scale_delta
            return amin, amax, poly_factors, q0

        def _apply(params, q):
            amin, amax, poly_factors, q0 = params
            q = q - q0
            value = _apply_poly_factors(poly_factors, q)
            return restrict(value, mn, mx, amin, amax, method=method)

        if center:

            def apply(params, q):
                return _apply(params, q) - _apply(params, flexion_center)

        else:
            apply = _apply

        return DrawnFnPair(init, apply)

    return factory


def ConstantValue_DrawnFnPair(value: float) -> DrawnFnPairFactory:
    value = jnp.array(value)

    def factory(xs, mn, mx):

        def init(key):
            return {}

        def apply(params, q):
            return value

        return DrawnFnPair(init, apply)

    return factory


def MLP_DrawnFnPair(
    center: bool = False, flexion_center: Optional[float] = None
) -> DrawnFnPairFactory:

    def factory(xs, mn, mx):
        nonlocal flexion_center

        flexion_mn = jnp.min(xs)
        flexion_mx = jnp.max(xs)

        if flexion_center is None:
            flexion_center = (flexion_mn + flexion_mx) / 2
        else:
            flexion_center = jnp.array(flexion_center)

        @hk.without_apply_rng
        @hk.transform
        def mlp(x):
            # normalize the x input; [0, 1]
            x = _shift(x, flexion_mn, flexion_mx)
            # center the x input; [-0.5, 0.5]
            x = x - 0.5
            net = hk.nets.MLP(
                [10, 5, 1],
                activation=jnp.tanh,
                w_init=hk.initializers.RandomNormal(),
            )
            return net(x)

        example_q = jnp.zeros((1,))

        def init(key):
            return mlp.init(key, example_q)

        def _apply(params, q):
            q = q[None]
            return jnp.squeeze(_shift_inv(jax.nn.sigmoid(mlp.apply(params, q)), mn, mx))

        if center:

            def apply(params, q):
                return _apply(params, q) - _apply(params, flexion_center)

        else:
            apply = _apply

        return DrawnFnPair(init, apply)

    return factory


def _gp_draw_and_rom(key, xs, ys, length_scale, mn, mx, amin, amax):
    randomized_ys = _gp_draw(key, xs, ys, length_scale)
    if ys is not None:
        amin += jnp.min(ys)
        amax += jnp.max(ys)
    return restrict(randomized_ys, mn, mx, amin, amax)


def _gp_draw(key, xs, ys=None, length: float = 1.0, noise=0.0, method="svd", **kwargs):
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
