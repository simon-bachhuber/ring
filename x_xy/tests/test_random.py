import pytest
from jax import random as jrand

from x_xy.random import random_angle_over_time, random_position_over_time


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
        T = 30
        ANG_0 = 0.0
        angle = random_angle_over_time(
            jrand.PRNGKey(1),
            jrand.PRNGKey(2),
            ANG_0,
            0.1,
            0.5,
            0.1,
            0.5,
            T,
            Ts,
            randomized_interpolation,
            range_of_motion,
            range_of_motion_method,
        )
        assert angle.shape == (int(T / Ts),)
        # TODO Why does this fail for ANG_0 != 0.0?
        assert angle[0] == ANG_0


def test_position():
    for Ts in [0.1, 0.01]:
        T = 30
        POS_0 = 0.0
        pos = random_position_over_time(
            jrand.PRNGKey(1), POS_0, -0.2, 0.2, 0.1, 0.5, 0.1, 0.5, T, Ts, 10
        )
        assert pos.shape == (int(T / Ts),)
        # TODO Why does this fail for POS_0 != 0.0?
        assert pos[0] == POS_0
