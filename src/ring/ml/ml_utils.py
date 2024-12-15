from collections import defaultdict
from functools import partial
import os
from pathlib import Path
import pickle
import shutil
import time
from typing import Optional, Protocol
import warnings

import jax
import numpy as np
from tree_utils import PyTree

import ring
import wandb

# An arbitrarily nested dictionary with Array leaves; Or strings
NestedDict = PyTree
STEP_METRIC_NAME = "i_episode"


class Logger(Protocol):
    def close(self) -> None: ...  # noqa: E704

    def log(self, metrics: NestedDict) -> None: ...  # noqa: E704

    @staticmethod
    def n_params(params) -> int:
        "Number of parameters in Pytree `params`."
        return sum([arr.flatten().size for arr in jax.tree_util.tree_leaves(params)])


class MixinLogger(Logger):
    def close(self):
        pass

    def log_image(self, path: str, caption: Optional[str] = None):
        raise NotImplementedError

    def log_video(
        self,
        path: str,
        fps: int = 25,
        caption: Optional[str] = None,
        step: Optional[int] = None,
    ):
        raise NotImplementedError

    def log_params(self, path: str):
        raise NotImplementedError

    def log(self, metrics: NestedDict):
        step = metrics[STEP_METRIC_NAME] if STEP_METRIC_NAME in metrics else None
        for key, value in _flatten_convert_filter_nested_dict(metrics).items():
            self.log_key_value(key, value, step=step)

    def log_key_value(self, key: str, value: str | float, step: Optional[int] = None):
        raise NotImplementedError

    def log_command_output(self, command: str):
        path = command.replace(" ", "_") + ".txt"
        os.system(f"{command} >> {path}")
        self.log_txt(path, wait=True)
        os.system(f"rm {path}")

    def log_txt(self, path: str, wait: bool = True):
        raise NotImplementedError

    def _log_environment(self):
        self.log_command_output("pip list")
        self.log_command_output("conda list")
        self.log_command_output("nvidia-smi")


class DictLogger(MixinLogger):
    def __init__(self, output_path: Optional[str] = None):
        self._logs = defaultdict(lambda: [])
        self._output_path = output_path

    def log_key_value(self, key: str, value: str | float, step: int | None = None):
        self._logs[key].append(value)

    def close(self):
        if self._output_path is None:
            return
        self.save(self._output_path)

    def save(self, path: str):
        path = Path(path).with_suffix(".pickle").expanduser()
        path.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(self.get_logs(), file, protocol=5)

    def get_logs(self):
        return self._logs


class WandbLogger(MixinLogger):
    def __init__(self):
        self._log_environment()
        wandb.run.define_metric(STEP_METRIC_NAME)

    def log_key_value(self, key: str, value: str | float, step: Optional[int] = None):
        data = {key: value}
        if step is not None:
            data.update({STEP_METRIC_NAME: step})
        wandb.log(data)

    def log_params(self, path: str):
        self.wandb_save(path)

    def log_video(
        self,
        path: str,
        fps: int = 25,
        caption: Optional[str] = None,
        step: Optional[int] = None,
    ):
        # TODO >>>
        self.wandb_save(path)
        return
        # <<<
        data = {"video": wandb.Video(path, caption=caption, fps=fps)}
        if step is not None:
            data.update({STEP_METRIC_NAME: step})
        wandb.log(data)

    def log_image(self, path: str, caption: Optional[str] = None):
        # wandb.log({"image": wandb.Image(path, caption=caption)})
        self.wandb_save(path)

    def log_txt(self, path: str, wait: bool = True):
        self.wandb_save(path)
        # TODO: `wandb` is not async at all?
        if wait:
            time.sleep(3)

    def close(self):
        wandb.run.finish()

    @staticmethod
    def wandb_save(path):
        if wandb.run is not None and wandb.run.settings._offline:
            # Create a dedicated directory in the WandB run directory to store copies
            # of files
            destination_dir = os.path.join(wandb.run.dir, "copied_files")
            os.makedirs(destination_dir, exist_ok=True)

            # Copy the file to this new location
            copied_file_path = os.path.join(destination_dir, os.path.basename(path))
            shutil.copy2(path, copied_file_path)

            # Use wandb.save to save the copied file (now a true copy)
            wandb.save(copied_file_path)
        else:
            wandb.save(path, policy="now")


def _flatten_convert_filter_nested_dict(
    metrices: NestedDict, filter_nan_inf: bool = True
):
    metrices = _flatten_dict(metrices)
    metrices = jax.tree_map(_to_float_if_not_string, metrices)

    if not filter_nan_inf:
        return metrices

    filtered_metrices = {}
    for key, value in metrices.items():
        if not isinstance(value, str) and (np.isnan(value) or np.isinf(value)):
            warning = f"Warning: Value of metric {key} is {value}. We skip it."
            warnings.warn(warning)
            continue
        filtered_metrices[key] = value
    return filtered_metrices


def _flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        k = str(k) if isinstance(k, int) else k
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _to_float_if_not_string(value):
    if isinstance(value, str):
        return value
    else:
        return float(value)


def on_cluster() -> bool:
    """Return `true` if executed on cluster."""
    env_var = os.environ.get("ON_CLUSTER", None)
    return False if env_var is None else True


_unique_id_is_logged = False


def unique_id() -> str:
    global _unique_id_is_logged
    if wandb.run is not None and _unique_id_is_logged is False:
        wandb.config["unique_id"] = ring._UNIQUE_ID
        _unique_id_is_logged = True
    return ring._UNIQUE_ID


def save_model_tf(jax_func, path: str, *input, validate: bool = True):
    from jax.experimental import jax2tf
    import tensorflow as tf

    signature = jax.tree_map(
        lambda arr: tf.TensorSpec(list(arr.shape), tf.float32), input
    )

    tf_func = jax2tf.convert(jax_func, with_gradient=False)

    class RingTFModule(tf.Module):
        @partial(
            tf.function, autograph=False, jit_compile=True, input_signature=signature
        )
        def __call__(self, *args):
            return tf_func(*args)

    model = RingTFModule()

    tf.saved_model.save(
        model,
        path,
        options=tf.saved_model.SaveOptions(experimental_custom_gradients=False),
        signatures={"default": model.__call__},
    )

    if validate:
        output_jax = jax_func(*input)
        output_tf = tf.saved_model.load(path)(*input)
        jax.tree_map(
            lambda a1, a2: np.allclose(a1, a2, atol=1e-5, rtol=1e-5),
            output_jax,
            output_tf,
        )


def to_onnx(
    fn,
    output_path,
    *args: tuple[np.ndarray],
    in_args_names: Optional[list[str]] = None,
    out_args_names: Optional[list[str]] = None,
    validate: bool = False,
):
    """
    Converts a JAX function to ONNX format, with optional input/output renaming and validation.

    Args:
        fn (callable): The JAX function to be converted.
        output_path (str): Path where the ONNX model will be saved.
        *args (tuple[np.ndarray]): Input arguments for the JAX function.
        in_args_names (Optional[list[str]]): Names for the ONNX model's input tensors. Defaults to None.
        out_args_names (Optional[list[str]]): Names for the ONNX model's output tensors. Defaults to None.
        validate (bool): Whether to validate the ONNX model against the JAX function's outputs. Defaults to False.

    Raises:
        AssertionError: If the number of provided names does not match the number of inputs/outputs.
        AssertionError: If the ONNX model's outputs do not match the JAX function's outputs within tolerance.
        ValueError: If any error occurs during ONNX conversion, saving, or validation.

    Notes:
        - The function uses `jax2tf` to convert the JAX function to TensorFlow format,
          and `tf2onnx` for ONNX conversion.
        - Input and output tensor names in the ONNX model can be renamed using `sor4onnx.rename`.
        - Validation compares outputs of the JAX function and the ONNX model using ONNX Runtime.

    Example:
        ```
        import jax.numpy as jnp

        def my_fn(x, y):
            return x + y, x * y

        x = jnp.array([1, 2, 3])
        y = jnp.array([4, 5, 6])

        to_onnx(
            my_fn,
            "model.onnx",
            x, y,
            in_args_names=["input1", "input2"],
            out_args_names=["sum", "product"],
            validate=True,
        )
        ```
    """  # noqa: E501
    import jax.experimental.jax2tf as jax2tf
    import tensorflow as tf
    import tf2onnx

    tf_fn = tf.function(jax2tf.convert(fn, enable_xla=False))
    tf_args = [tf.TensorSpec(np.shape(x), np.result_type(x)) for x in args]
    tf2onnx.convert.from_function(
        tf_fn, input_signature=tf_args, output_path=output_path
    )

    if in_args_names is not None or out_args_names is not None:
        import onnx
        from sor4onnx import rename

        model = onnx.load(output_path)

        if in_args_names is not None:
            old_names = [inp.name for inp in model.graph.input]
            assert len(old_names) == len(in_args_names)
            for old_name, new_name in zip(old_names, in_args_names):
                model = rename([old_name, new_name], None, model, None, mode="inputs")

        if out_args_names is not None:
            old_names = [out.name for out in model.graph.output]
            assert len(old_names) == len(out_args_names)
            for old_name, new_name in zip(old_names, out_args_names):
                model = rename([old_name, new_name], None, model, None, mode="outputs")

        onnx.save(model, output_path)

    if validate:
        import onnxruntime as ort

        output_jax = fn(*args)
        session = ort.InferenceSession(output_path)
        input_names = [inp.name for inp in session.get_inputs()]
        output_onnx = session.run(
            None, {name: np.array(arg) for name, arg in zip(input_names, args)}
        )

        for o1, o2 in zip(output_jax, output_onnx):
            assert np.allclose(o1, o2, atol=1e-5, rtol=1e-5)

        if out_args_names is not None:
            assert [out.name for out in session.get_outputs()] == out_args_names


def _unknown_link_names(N: int):
    return [f"link{i}" for i in range(N)]
