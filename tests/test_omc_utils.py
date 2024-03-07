import numpy as np
import qmt
from ring.exp import omc_utils as omc


def quatfromangles(angles: np.ndarray):
    quats = []
    for angle in angles:
        quats.append(qmt.quatFromAngleAxis(angle, np.array([1.0, 0, 0])))
    return np.vstack(quats)


def test_resample():
    arange_twice_freq = np.array(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            5.5,
            6.0,
            6.5,
            7.0,
            7.5,
            8.0,
            8.5,
            9.0,
            9.0,
        ]
    )
    signal_in = {
        "vector": np.arange(10),
        "quat": quatfromangles(np.arange(10)),
    }

    signal_out = omc.resample(signal_in, 1, 2)

    np.testing.assert_allclose(signal_out["vector"], arange_twice_freq)
    np.testing.assert_allclose(
        signal_out["quat"],
        quatfromangles(arange_twice_freq),
    )

    arange_less_freq = np.array([0.0, 1.25, 2.5, 3.75, 5.0, 6.25, 7.5, 8.75])

    signal_out = omc.resample(signal_in, 1, 0.8)

    np.testing.assert_allclose(signal_out["vector"], arange_less_freq)
    np.testing.assert_allclose(
        signal_out["quat"],
        quatfromangles(arange_less_freq),
    )

    signal_in = {
        "not_a_quat": np.hstack(tuple(np.arange(10)[:, None] for _ in range(4)))
    }

    signal_out = omc.resample(signal_in, 1, 2, quatdetect=False)

    np.testing.assert_allclose(
        signal_out["not_a_quat"],
        np.hstack(tuple(arange_twice_freq[:, None] for _ in range(4))),
    )


def test_crop_tail():
    data = {"gyr": np.arange(100), "omc": np.arange(100)}

    data_cropped = omc.crop_tail(data, {"gyr": 40.0, "omc": 30.0})
    np.testing.assert_allclose(data_cropped["gyr"], data["gyr"])
    np.testing.assert_allclose(data_cropped["omc"], data["omc"][:75])

    data_cropped = omc.crop_tail(data, {"gyr": 40.0, "omc": 100.0})
    np.testing.assert_allclose(data_cropped["gyr"], data["gyr"][:40])
    np.testing.assert_allclose(data_cropped["omc"], data["omc"])

    data_cropped = omc.crop_tail(data, {"gyr": 40.0, "omc": 120.0})
    np.testing.assert_allclose(data_cropped["gyr"], data["gyr"][:30])
    np.testing.assert_allclose(data_cropped["omc"], data["omc"][:90])


def test_resample_and_crop_tail():
    data = {"gyr": np.arange(100.0), "omc": np.arange(105, step=1 / 3)}
    data_resampled = omc.resample(data, {"gyr": 40.0, "omc": 120.0}, 100.0)
    data_cropped = omc.crop_tail(data_resampled)

    np.testing.assert_allclose(
        data_cropped["gyr"][:-2], np.arange(100.0, step=1 / 2.5)[:-2]
    )
    np.testing.assert_allclose(
        data_cropped["omc"][:-2], np.arange(100.0, step=1 / 2.5)[:-2]
    )
