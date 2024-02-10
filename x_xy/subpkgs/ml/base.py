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
    def _predict_2d(self, X: dict, sys: x_xy.System | None, **kwargs) -> dict:
        X = tree_utils.add_batch_dim(X)
        y = self._predict_3d(X, sys, **kwargs)
        return tree_utils.tree_slice(y, 0)

    @abstractmethod
    def _predict_3d(self, X: dict, sys: x_xy.System | None, **kwargs) -> dict:
        pass

    def predict(self, X: dict, sys: x_xy.System | None, **kwargs) -> dict:
        "X.shape = (bs, timesteps, features) or (timesteps, features)"
        if tree_utils.tree_ndim(X) == 2:
            return self._predict_2d(X, sys, **kwargs)
        else:
            return self._predict_3d(X, sys, **kwargs)

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
    def _predict_2d(self, X: dict, sys: System | None, **kwargs) -> dict:
        pass

    def _predict_3d(self, X: dict, sys: System | None, **kwargs) -> dict:
        N = tree_utils.tree_shape(X)
        ys = []
        for i in range(N):
            ys.append(self._predict_2d(tree_utils.tree_slice(X, i), sys, **kwargs))
        return tree_utils.tree_batch(ys)


class InitApplyFnFilter(AbstractFilter):
    def __init__(
        self,
        init_apply_fn_factory: Callable[[x_xy.System | None], SimpleNamespace],
        name: Optional[str] = None,
        params: Optional[str | tree_utils.PyTree] = None,
        key: jax.Array = jax.random.PRNGKey(1),
        lpf: Optional[float] = None,
        X_transform=None,
    ):
        self._name = name
        self.key = key
        self.params = self._load_params(params)
        self.init_apply_fn = None
        self.init_apply_fn_factory = init_apply_fn_factory
        self.lpf = lpf
        self.X_transform = X_transform

    def _predict_3d(
        self,
        X: dict,
        sys: System | None,
        params: dict | None = None,
        state: dict | None = None,
    ) -> dict:

        if sys is not None:
            self.set_sys(sys)

        if (params is None and self.params is None) or (state is None):
            _params, _state = self.init_apply_fn.init(self.key, X)

        if params is not None:
            pass
        elif self.params is not None:
            params = self.params
        else:
            params = _params

        if state is not None:
            pass
        else:
            state = _state
            bs = tree_utils.tree_shape(X)
            state = jax.tree_map(lambda arr: jnp.repeat(arr[None], bs, axis=0), state)

        if self.X_transform is not None:
            X = self.X_transform(X)

        yhat = self.init_apply_fn.apply(params, state, X)[0]

        if self.lpf is not None:
            yhat = jax.tree_map(
                jax.vmap(
                    lambda q: x_xy.maths.quat_lowpassfilter(q, self.lpf, filtfilt=True)
                ),
                yhat,
            )

        return yhat

    def set_sys(self, sys):
        self.init_apply_fn = self.init_apply_fn_factory(sys)

    def get_state(self, X, bs: int):
        _, state = self.init_apply_fn.init(self.key, X)
        state = jax.tree_map(lambda arr: jnp.repeat(arr[None], bs, axis=0), state)
        return state

    @staticmethod
    def _load_params(params: str | tree_utils.PyTree | None):
        if params is None:
            return None
        if isinstance(params, str):
            return load(params)
        return params
