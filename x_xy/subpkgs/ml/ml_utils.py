from abc import ABC
from abc import abstractmethod
from abc import abstractstaticmethod
from collections import namedtuple
from functools import partial
import logging
import os
from pathlib import Path
import pickle
import time
from typing import Optional, Union
import webbrowser

import jax
import numpy as np
from tree_utils import PyTree
from tree_utils import tree_batch

import wandb
import x_xy
from x_xy.io import load_sys_from_str
from x_xy.utils import download_from_repo
from x_xy.utils import import_lib

suffix = ".pickle"


def save(data: PyTree, path: Union[str, Path], overwrite: bool = False):
    path = Path(path).expanduser()
    if path.suffix != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f"File {path} already exists.")
    with open(path, "wb") as file:
        pickle.dump(data, file, protocol=5)


def load(
    path: Optional[Union[str, Path]] = None,
    pretrained: Optional[str] = None,
    pretrained_version: Optional[int] = None,
) -> PyTree:
    assert not (
        path is None and pretrained is None
    ), "Either `path` or `pretrained` must be given."
    assert not (
        path is not None and pretrained is not None
    ), "Both `path` and `pretrained` cannot both be given."

    if pretrained_version is not None:
        assert pretrained is not None

    if path is not None:
        path = Path(path).expanduser()
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        if path.suffix != suffix:
            raise ValueError(f"Not a {suffix} file: {path}")
        with open(path, "rb") as file:
            data = pickle.load(file)
        return data
    else:
        version = ""
        if pretrained_version is not None:
            # v0, v1, v2, ...
            version = f"_v{int(pretrained_version)}"
        path_in_repo = f"params/{pretrained}/params_{pretrained}{version}{suffix}"
        path_on_disk = download_from_repo(path_in_repo)
        return load(path_on_disk)


def list_pretrained() -> None:
    "Open Github repo that hosts the pretrained parameters."
    url = "https://github.com/SimiPixel/x_xy_v2_datahost/tree/main/params"
    webbrowser.open(url)


# An arbitrarily nested dictionary with jax.Array leaves; Or strings
NestedDict = PyTree
STEP_METRIC_NAME = "i_episode"


class Logger(ABC):
    @abstractmethod
    def log(self, metrics: NestedDict):
        pass

    def close(self):
        pass


def n_params(params) -> int:
    "Number of parameters in Pytree `params`."
    return sum([arr.flatten().size for arr in jax.tree_util.tree_leaves(params)])


class DictLogger(Logger):
    def __init__(self, output_path: Optional[str] = None):
        self._logs = {}
        self._output_path = output_path

    def log(self, metrics: NestedDict):
        metrics = _flatten_convert_filter_nested_dict(metrics, filter_nan_inf=False)
        metrics = tree_batch([metrics])

        for key in metrics:
            existing_keys = []
            if key in self._logs:
                existing_keys.append(key)
            else:
                self._logs[key] = metrics[key]

        if len(existing_keys) > 0:
            self._logs.update(
                tree_batch(
                    [
                        {key: self._logs[key] for key in existing_keys},
                        {key: metrics[key] for key in existing_keys},
                    ],
                    True,
                )
            )

    def close(self):
        if self._output_path is None:
            return
        self.save(self._output_path)

    def save(self, path: str):
        path = Path(path).with_suffix(suffix).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(self._logs, file, protocol=5)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as file:
            logs = pickle.load(file)
        logger = DictLogger(path)
        logger._logs = logs
        return DictLogger


class MultimediaLogger(Logger):
    @abstractmethod
    def log_image(self, path: str, caption: Optional[str] = None):
        pass

    @abstractmethod
    def log_video(
        self,
        path: str,
        fps: int = 25,
        caption: Optional[str] = None,
        step: Optional[int] = None,
    ):
        pass

    @abstractmethod
    def log_params(self, path: str):
        pass

    def log(self, metrics: NestedDict):
        step = metrics[STEP_METRIC_NAME] if STEP_METRIC_NAME in metrics else None
        for key, value in _flatten_convert_filter_nested_dict(metrics).items():
            self.log_key_value(key, value, step=step)

    @abstractmethod
    def log_key_value(self, key: str, value: str | float, step: Optional[int] = None):
        pass

    def log_command_output(self, command: str):
        path = command.replace(" ", "_") + ".txt"
        os.system(f"{command} >> {path}")
        self.log_txt(path, wait=True)
        os.system(f"rm {path}")

    @abstractmethod
    def log_txt(self, path: str, wait: bool = True):
        pass

    @staticmethod
    def _print_upload_file(path: str):
        logging.info(f"Uploading file {path}.")

    @abstractstaticmethod
    def disable():
        pass


class MockMultimediaLogger(MultimediaLogger):
    def log_image(self, path: str, caption: str | None = None):
        pass

    def log_video(
        self,
        path: str,
        fps: int = 25,
        caption: str | None = None,
        step: int | None = None,
    ):
        pass

    def log_params(self, path: str):
        pass

    def log_key_value(self, key: str, value: str | float, step: int | None = None):
        pass

    def log_txt(self, path: str, wait: bool = True):
        pass

    @staticmethod
    def disable():
        pass


class NeptuneLogger(MultimediaLogger):
    def __init__(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """Logger that logs the training progress to Neptune.

        Args:
            project (Optional[str], optional): Name of the project where the run should
                go, in the form "workspace-name/project_name". Can also be provided
                using the environemnt variable `NEPTUNE_PROJECT`
            name (Optional[str], optional): Identifier inside the project. Can also be
                provided using the environment variable `NEPTUNE_NAME`

        Raises:
            Exception: If environment variable `NEPTUNE_TOKEN` is unset.
        """
        api_token = os.environ.get("NEPTUNE_TOKEN", None)
        if api_token is None:
            raise Exception(
                "Could not find the token for neptune logging. Make sure that the \
                            environment variable `NEPTUNE_TOKEN` is set."
            )

        if name is None:
            name = os.environ.get("NEPTUNE_NAME", None)

        neptune = import_lib("neptune")

        self.run = neptune.init_run(
            name=name,
            project=project,
            api_token=api_token,
        )

        _log_environment(self)

    def log_key_value(self, key: str, value: str | float, step: Optional[int] = None):
        self.run[key].log(value)

    def log_params(self, path: str):
        self._print_upload_file(path)
        # if we wouldn't wait then the run might end before upload finishes
        self.run[f"params/{_file_name(path, extension=True)}"].upload(path, wait=True)

    def log_video(
        self,
        path: str,
        fps: int = 25,
        caption: Optional[str] = None,
        step: Optional[int] = None,
    ):
        self.run[f"video/{_file_name(path, extension=True)}"].upload(path)

    def log_image(self, path: str, caption: Optional[str] = None):
        self.run[f"image/{_file_name(path, extension=True)}"].upload(path)

    def log_txt(self, path: str, wait: bool = True):
        self.run[f"txt/{_file_name(path)}"].upload(path, wait=wait)

    def close(self):
        self.run.stop()

    @staticmethod
    def disable():
        os.environ["NEPTUNE_MODE"] = "debug"


class WandbLogger(MultimediaLogger):
    def __init__(self):
        _log_environment(self)
        wandb.run.define_metric(STEP_METRIC_NAME)

    def log_key_value(self, key: str, value: str | float, step: Optional[int] = None):
        data = {key: value}
        if step is not None:
            data.update({STEP_METRIC_NAME: step})
        wandb.log(data)

    def log_params(self, path: str):
        self._print_upload_file(path)
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

    @staticmethod
    def disable():
        os.environ["WANDB_MODE"] = "offline"

    def close(self):
        wandb.run.finish()


def disable_syncing_to_cloud():
    NeptuneLogger.disable()
    WandbLogger.disable()


def _file_name(path: str, extension: bool = False):
    file = path.split("/")[-1]
    return file if extension else file.split(".")[0]


def _log_environment(logger: MultimediaLogger):
    logger.log_command_output("pip list")
    logger.log_command_output("conda list")
    logger.log_command_output("nvidia-smi")


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
            print(f"Warning: Value of metric {key} is {value}. We skip it.")
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
    return x_xy._UNIQUE_ID


InitApplyFnPair = namedtuple("InitApplyFnPair", ["init", "apply"])


_DUMMY_BODY_NAME = "global"
_dummy_sys_xml_str = f"""
<x_xy model="free">
    <worldbody>
        <body name="{_DUMMY_BODY_NAME}" joint="frozen"></body>
    </worldbody>
</x_xy>
"""


def make_non_social_version(make_social_version, kwargs: dict):
    kwargs["sys"] = load_sys_from_str(_dummy_sys_xml_str)
    kwargs["keep_toRoot_output"] = True

    output_transform = kwargs.get("link_output_transform", None)
    if output_transform is not None:

        def _wrapped_transform(y):
            y_pytree = output_transform(y)
            return {0: y_pytree}

        kwargs["link_output_transform"] = _wrapped_transform

    dummy_rnno = make_social_version(**kwargs)

    def non_social_init(key, X):
        return dummy_rnno.init(key, {_DUMMY_BODY_NAME: X})

    def non_social_apply(params, state, X):
        yhat, state = dummy_rnno.apply(params, state, {_DUMMY_BODY_NAME: X})
        return yhat[_DUMMY_BODY_NAME], state

    return InitApplyFnPair(init=non_social_init, apply=non_social_apply)


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


def model_wrapper_indices_to_names(
    init_apply_pair: InitApplyFnPair, link_names: list[str]
) -> InitApplyFnPair:
    def apply(params, state, *args):
        out, state = init_apply_pair.apply(params, state, *args)
        out_names = dict()
        for key, val in out.items():
            out_names[link_names[key]] = val
        return out_names, state

    return InitApplyFnPair(init=init_apply_pair.init, apply=apply)
