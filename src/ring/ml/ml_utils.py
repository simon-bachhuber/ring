from collections import defaultdict
from functools import partial
import os
from pathlib import Path
import pickle
import time
from typing import Optional, Protocol
import warnings

import jax
import numpy as np
from tree_utils import PyTree

import ring
from ring.utils import import_lib
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
        wandb.save(path, policy="now")

    def log_video(
        self,
        path: str,
        fps: int = 25,
        caption: Optional[str] = None,
        step: Optional[int] = None,
    ):
        # TODO >>>
        wandb.save(path, policy="now")
        return
        # <<<
        data = {"video": wandb.Video(path, caption=caption, fps=fps)}
        if step is not None:
            data.update({STEP_METRIC_NAME: step})
        wandb.log(data)

    def log_image(self, path: str, caption: Optional[str] = None):
        # wandb.log({"image": wandb.Image(path, caption=caption)})
        wandb.save(path, policy="now")

    def log_txt(self, path: str, wait: bool = True):
        wandb.save(path, policy="now")
        # TODO: `wandb` is not async at all?
        if wait:
            time.sleep(3)

    def close(self):
        wandb.run.finish()


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


def unique_id() -> str:
    return ring._UNIQUE_ID


def save_model_tf(jax_func, path: str, *input, validate: bool = True):
    from jax.experimental import jax2tf

    tf = import_lib("tensorflow", "the function `save_model_tf`")

    def _create_module(jax_func, input):
        signature = jax.tree_map(
            lambda arr: tf.TensorSpec(list(arr.shape), tf.float32), input
        )

        class RingTFModule(tf.Module):
            def __init__(self, jax_func):
                super().__init__()
                self.tf_func = jax2tf.convert(jax_func, with_gradient=False)

            @partial(
                tf.function,
                autograph=False,
                jit_compile=True,
                input_signature=signature,
            )
            def __call__(self, *args):
                return self.tf_func(*args)

        return RingTFModule(jax_func)

    model = _create_module(jax_func, input)
    tf.saved_model.save(
        model,
        path,
        options=tf.saved_model.SaveOptions(experimental_custom_gradients=False),
    )
    if validate:
        output_jax = jax_func(*input)
        output_tf = tf.saved_model.load(path)(*input)
        jax.tree_map(
            lambda a1, a2: np.allclose(a1, a2, atol=1e-5, rtol=1e-5),
            output_jax,
            output_tf,
        )


def _unknown_link_names(N: int):
    return [f"link{i}" for i in range(N)]
