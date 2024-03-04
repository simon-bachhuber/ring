from pathlib import Path

import numpy as np

_onnx_path = Path(__file__).parent.joinpath("riann.onnx")


class RIANN:
    def __init__(self):
        import onnxruntime as rt

        self.session = rt.InferenceSession(_onnx_path)
        self.h0 = np.zeros((2, 1, 1, 200), dtype=np.float32)

    def predict(self, acc, gyr, fs):
        """
        Update plot with external x-values.
        Parameters
        ----------
        acc: numpy-array [sequence_length x 3]
            Acceleration data of the IMU. The axis order is x,y,z.
        gyr: numpy-array [sequence_length x 3]
            Gyroscope data of the IMU. The axis order is x,y,z.
        fs: float
            Samplingrate of the provided IMU data

        Returns
        -------
        attitude unit-quaternions [sequence_length x 4]
        """
        # prepare minibatch for runtime execution
        np_inp = np.concatenate(
            [acc, gyr, np.tile(1 / fs, (acc.shape[0], 1))], axis=-1
        ).astype(np.float32)[None, ...]

        return self.session.run([], {"input": np_inp, "h0": self.h0})[0][0]
