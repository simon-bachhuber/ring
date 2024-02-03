from typing import Optional

import jax
import numpy as np
import qmt

import x_xy
from x_xy.subpkgs import ml
from x_xy.subpkgs.ml.riann import RIANN


def _riann_predict(gyr, acc, params: dict):
    fs = 1 / params["Ts"]
    riann = RIANN()
    return riann.predict(acc, gyr, fs)


_attitude_methods = {
    "vqf": ("VQFAttitude", qmt.oriEstVQF),
    "madgwick": ("MadgwickAttitude", qmt.oriEstMadgwick),
    "mahony": ("MahonyAttitude", qmt.oriEstMahony),
    "seel": ("SeelAttitude", qmt.oriEstIMU),
    "riann": ("RIANN", _riann_predict),
}


class AttitudeBaseline(ml.AbstractFilter2d):
    def __init__(self, method: str) -> None:
        """ """
        self._name, self._method = _attitude_methods[method]

    def _predict_2d(self, X, sys):
        quats = dict()
        for name, imu_data in X.items():
            quats[name] = self._method(
                gyr=imu_data["gyr"], acc=imu_data["acc"], params=dict(Ts=float(sys.dt))
            )

        # NOTE CONVENTION !!
        quats = {name: qmt.qinv(quats[name]) for name in quats}
        return quats


class TwoSeg1D(ml.AbstractFilter2d):
    def __init__(
        self,
        method: str,
        seg: str,
        hinge_joint_axis: np.ndarray | None = None,
        lpf_glo: Optional[float] = None,
        lpf_rel: Optional[float] = None,
    ):
        """Estimates Inclination and Relative Pose for Hinge Joint. Uses 6D VQF.

        Args:
            hinge_joint_axis (np.ndarray): Known Hinge Joint Axis.
            name (str): Name of filter
            first_seg_name (str): First segment name.
            second_seg_name (str): Second segment name.
        """

        self.hinge_joint_axis = None
        if hinge_joint_axis is not None:
            assert hinge_joint_axis.shape == (3,)
            self.hinge_joint_axis = hinge_joint_axis
        self.seg = seg
        self.method = method
        assert method in ["euler_2d", "euler_1d", "1d_corr", "proj"]
        name = method + f"_ollson_{int(hinge_joint_axis is None)}"
        self._name = name
        self.lpf_glo, self.lpf_rel = lpf_glo, lpf_rel

    def _predict_2d(self, X, sys):
        X = jax.tree_map(lambda arr: np.asarray(arr).copy(), X)

        assert len(X) == 2
        Ts = float(sys.dt)

        # VQF
        quats = dict()
        for name, imu_data in X.items():
            quats[name] = qmt.oriEstVQF(
                imu_data["gyr"], imu_data["acc"], params=dict(Ts=Ts)
            )

        # heading correction of second orientation estimate
        first, second = self.seg, list(set(X.keys()) - set([self.seg]))[0]
        gyr1, gyr2 = X[first]["gyr"], X[second]["gyr"]

        t = np.arange(gyr1.shape[0] * Ts, step=Ts)

        if self.hinge_joint_axis is None:
            # returns jhat1, jhat2 which both are shape = (3x1)
            self.hinge_joint_axis = qmt.jointAxisEstHingeOlsson(
                X[first]["acc"],
                X[second]["acc"],
                gyr1,
                gyr2,
                estSettings=dict(quiet=True),
            )[0][:, 0]

        quats[second] = qmt.headingCorrection(
            gyr1,
            gyr2,
            quats[first],
            quats[second],
            t,
            self.hinge_joint_axis,
            None,
            estSettings=dict(constraint=self.method),
        )[0]

        # NOTE CONVENTION !!
        quats = {name: qmt.qinv(quats[name]) for name in quats}

        quats = _maybe_lowpassfilter_quats(quats, self.lpf_glo)

        yhat = dict()

        # tibia to femur
        yhat[second] = x_xy.maths.quat_mul(
            quats[first], x_xy.maths.quat_inv(quats[second])
        )
        # add it such that it gets low-pass-filtered for `lpf_rel` too
        yhat[first] = quats[first]
        yhat = _maybe_lowpassfilter_quats(yhat, self.lpf_rel)

        return yhat


class NSeg3D_9DVQF(ml.AbstractFilter2d):
    def __init__(
        self,
        name: str,
        chain: list[str],
        lpf_glo: Optional[float] = None,
        lpf_rel: Optional[float] = None,
    ):
        """Use 9D VQF on kinematic chain.

        Args:
            name (str): Name of filter
            chain (list[str]): Name of segments that make up chain.
        """
        self._name = name
        self.chain = chain
        self.lpf_glo, self.lpf_rel = lpf_glo, lpf_rel

    def _predict_2d(self, X, sys):
        del sys

        # VQF
        quats = dict()
        for name, imu_data in X.items():
            quats[name] = qmt.oriEstVQF(
                imu_data["gyr"], imu_data["acc"], imu_data["mag"], params=dict(Ts=0.01)
            )

        # NOTE CONVENTION !!
        quats = {name: qmt.qinv(quats[name]) for name in quats}

        quats = _maybe_lowpassfilter_quats(quats, self.lpf_glo)

        yhat = dict()

        # relative pose
        for i in range(1, len(self.chain)):
            first, second = self.chain[i - 1], self.chain[i]
            # tibia to femur
            yhat[second] = x_xy.maths.quat_mul(
                quats[first], x_xy.maths.quat_inv(quats[second])
            )

        # global aspect
        yhat[self.chain[0]] = quats[self.chain[0]]

        yhat = _maybe_lowpassfilter_quats(yhat, self.lpf_rel)

        return yhat


def _maybe_lowpassfilter_quats(quats: dict, cutoff_freq: float | None):
    if cutoff_freq is None:
        return quats
    return {
        name: x_xy.maths.quat_lowpassfilter(quats[name], cutoff_freq, filtfilt=True)
        for name in quats
    }
