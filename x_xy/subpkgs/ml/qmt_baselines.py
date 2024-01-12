from typing import Optional

import numpy as np
import qmt

import x_xy
from x_xy.subpkgs import ml


class TwoSeg1D(ml.AbstractFilter2d):
    def __init__(
        self,
        hinge_joint_axis: np.ndarray,
        name: str,
        first_seg_name: str,
        second_seg_name: str,
        lpf_glo: Optional[float] = None,
        lpf_rel: Optional[float] = None,
    ):
        """Estimates Inclination and Relative Pose for Hinge Joint.

        Args:
            hinge_joint_axis (np.ndarray): Known Hinge Joint Axis.
            name (str): Name of filter
            first_seg_name (str): First segment name.
            second_seg_name (str): Second segment name.
        """
        self.hinge_joint_axis = np.atleast_2d(hinge_joint_axis)[0]
        self._name = name
        self.first, self.second = first_seg_name, second_seg_name
        self.lpf_glo, self.lpf_rel = lpf_glo, lpf_rel

    def _predict_2d(self, X, sys):
        del sys

        # VQF
        quats = dict()
        for name, imu_data in X.items():
            quats[name] = qmt.oriEstVQF(
                imu_data["gyr"], imu_data["acc"], params=dict(Ts=0.01)
            )

        # heading correction of second orientation estimate
        first, second = self.first, self.second
        gyr1, gyr2 = X[first]["gyr"], X[second]["gyr"]
        t = np.arange(gyr1.shape[0] / 100, step=0.01)
        quats[second] = qmt.headingCorrection(
            gyr1, gyr2, quats[first], quats[second], t, self.hinge_joint_axis, None
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

        # inclination angle; eps to femur
        yhat[first] = x_xy.maths.quat_project(yhat[first], np.array([0.0, 0, 1]))[1]

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
