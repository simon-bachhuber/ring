from abc import ABC
from abc import abstractmethod
from types import SimpleNamespace
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import tree_utils

import x_xy
from x_xy.base import System

from .ml_utils import load


class AbstractFilter(ABC):
    def _predict_2d(self, X: dict, sys: x_xy.System | None) -> dict:
        X = tree_utils.add_batch_dim(X)
        y = self._predict_3d(X, sys)
        return tree_utils.tree_slice(y, 0)

    @abstractmethod
    def _predict_3d(self, X: dict, sys: x_xy.System | None) -> dict:
        pass

    def predict(self, X: dict, sys: x_xy.System | None) -> dict:
        "X.shape = (bs, timesteps, features) or (timesteps, features)"
        if tree_utils.tree_ndim(X) == 2:
            return self._predict_2d(X, sys)
        else:
            return self._predict_3d(X, sys)

    @property
    def name(self) -> str:
        if not hasattr(self, "_name"):
            raise NotImplementedError

        if self._name is None:
            raise RuntimeError("No `name` was given.")
        return self._name


class AbstractFilter2d(AbstractFilter):
    "Same as `AbstractFilter` but have to define `_predict_2d`"

    @abstractmethod
    def _predict_2d(self, X: dict, sys: System | None) -> dict:
        pass

    def _predict_3d(self, X: dict, sys: System | None) -> dict:
        N = tree_utils.tree_shape(X)
        ys = []
        for i in range(N):
            ys.append(self._predict_2d(tree_utils.tree_slice(X, i), sys))
        return tree_utils.tree_batch(ys)


class InitApplyFnFilter(AbstractFilter):
    def __init__(
        self,
        init_apply_fn_factory: Callable[[x_xy.System | None], SimpleNamespace],
        name: Optional[str] = None,
        params: Optional[str | tree_utils.PyTree] = None,
        key: jax.Array = jax.random.PRNGKey(1),
        lpf: Optional[float] = None,
    ):
        self._name = name
        self.key = key
        self.params = self._load_params(params)
        self.init_apply_fn_factory = init_apply_fn_factory
        self.lpf = lpf

    def _predict_3d(self, X: dict, sys: System | None) -> dict:
        init_apply_fn = self.init_apply_fn_factory(sys)
        params, state = init_apply_fn.init(self.key, X)
        params = params if self.params is None else self.params
        bs = tree_utils.tree_shape(X)
        state = jax.tree_map(lambda arr: jnp.repeat(arr[None], bs, axis=0), state)
        yhat = init_apply_fn.apply(params, state, X)[0]

        if self.lpf is not None:
            yhat = jax.tree_map(
                jax.vmap(
                    lambda q: x_xy.maths.quat_lowpassfilter(q, self.lpf, filtfilt=True)
                ),
                yhat,
            )

        return yhat

    @staticmethod
    def _load_params(params: str | tree_utils.PyTree | None):
        if params is None:
            return None
        if isinstance(params, str):
            return load(params)
        return params
