from jax import random as jrand
import numpy as np
import pytest

import ring
from ring.algorithms._random import _resolve_range_of_motion


@pytest.mark.parametrize(
    "range_of_motion, range_of_motion_method, delta_ang_min, delta_ang_max",
    [
        (False, "uniform", 0.0, 3.0),
        (True, "uniform", 2.0, 3.0),
        (True, "coinflip", 0.0, 0.1),
        (True, "uniform", 2.5, 3.0),
    ],
)
def test_delta_ang_min_max(
    range_of_motion, range_of_motion_method, delta_ang_min, delta_ang_max
):
    max_iter = 100
    t_min, t_max = 0.1, 3.1
    for seed in range(10):
        for prev_phi in [-3.0, 0.0, 3.0]:
            key_t, key_ang = jrand.split(jrand.PRNGKey(seed))
            dt, next_phi = _resolve_range_of_motion(
                range_of_motion,
                range_of_motion_method,
                -2 * np.pi,
                2 * np.pi,
                0.0,
                3.14,
                float(delta_ang_min),
                float(delta_ang_max),
                t_min,
                t_max,
                prev_phi,
                key_t,
                key_ang,
                max_iter,
            )
            assert t_min <= dt <= t_max
            assert delta_ang_min < abs(next_phi - prev_phi) < delta_ang_max


@pytest.mark.parametrize(
    "randomized_interpolation, range_of_motion, range_of_motion_method",
    [
        (False, False, "uniform"),
        (False, False, "coinflip"),
        (False, True, "uniform"),
        (False, True, "coinflip"),
        (True, False, "uniform"),
        (True, False, "coinflip"),
        (True, True, "uniform"),
        (True, True, "coinflip"),
    ],
)
def test_angle(randomized_interpolation, range_of_motion, range_of_motion_method):
    for Ts in [0.1, 0.01]:
        for ANG_0 in [0.0, -0.5, 0.5]:
            T = 30
            angle = ring.algorithms.random_angle_over_time(
                jrand.PRNGKey(1),
                jrand.PRNGKey(2),
                ANG_0,
                0.1,
                0.5,
                0.0,
                2 * np.pi,
                0.1,
                0.5,
                T,
                Ts,
                None,
                5,
                randomized_interpolation,
                range_of_motion,
                range_of_motion_method,
            )
            assert angle.shape == (int(T / Ts),)
            np.testing.assert_allclose(angle[0], ANG_0)


def test_position():
    for Ts in [0.1, 0.01]:
        T = 30
        for POS_0 in [0.0, 1.0]:
            pos = ring.algorithms.random_position_over_time(
                jrand.PRNGKey(1), POS_0, -0.2, 0.2, 0.1, 0.5, 0.1, 0.5, T, Ts, None, 10
            )
            assert pos.shape == (int(T / Ts),)
            # TODO Why does this fail for POS_0 != 0.0?
            assert pos[0] == POS_0
