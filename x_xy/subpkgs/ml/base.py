from abc import ABC
from abc import abstractmethod

import jax
import jax.numpy as jnp
import tree_utils

import x_xy
from x_xy.utils import pickle_load
from x_xy.utils import pickle_save


def _to_3d(tree):
    if tree is None:
        return None
    return jax.tree_map(lambda arr: arr[None], tree)


def _to_2d(tree, i: int = 0):
    if tree is None:
        return None
    return jax.tree_map(lambda arr: arr[i], tree)


class AbstractFilter(ABC):
    def _apply_unbatched(self, X, params, state, y, lam):
        return _to_2d(
            self._apply_batched(
                X=_to_3d(X), params=params, state=_to_3d(state), y=_to_3d(y), lam=lam
            )
        )

    @abstractmethod
    def _apply_batched(self, X, params, state, y, lam):
        pass

    @abstractmethod
    def init(self, bs, X, lam, seed: int):
        pass

    def apply(self, X, params=None, state=None, y=None, lam=None):
        "X.shape = (B, T, N, F) or (T, N, F)"
        assert X.ndim in [3, 4]
        if X.ndim == 4:
            return self._apply_batched(X, params, state, y, lam)
        else:
            return self._apply_unbatched(X, params, state, y, lam)

    @property
    def name(self) -> str:
        if not hasattr(self, "_name"):
            raise NotImplementedError

        if self._name is None:
            raise RuntimeError("No `name` was given.")
        return self._name

    def nojit(self) -> "AbstractFilter":
        return self

    def train(self) -> "AbstractFilter":
        return self

    def eval(self) -> "AbstractFilter":
        return self

    def _pre_save(self, *args, **kwargs) -> None:
        pass

    def save(self, path: str, *args, **kwargs):
        self._pre_save(*args, **kwargs)
        pickle_save(self.nojit(), path, overwrite=True)

    @staticmethod
    def _post_load(filter: "AbstractFilter", *args, **kwargs) -> "AbstractFilter":
        pass

    @classmethod
    def load(cls, path: str, *args, **kwargs):
        filter = pickle_load(path)
        return cls._post_load(filter, *args, **kwargs)


class AbstractFilterUnbatched(AbstractFilter):
    @abstractmethod
    def _apply_unbatched(self, X, params, state, y, lam):
        pass

    def _apply_batched(self, X, params, state, y, lam):
        N = X.shape[0]
        ys = []
        for i in range(N):
            ys.append(
                self._apply_unbatched(
                    _to_2d(X, i), params, _to_2d(state, i), _to_2d(y, i), lam
                )
            )
        return tree_utils.tree_batch(ys)


class AbstractFilterWrapper(AbstractFilter):
    def __init__(self, filter: AbstractFilter) -> None:
        self._filter = filter

    def _apply_batched(self, X, params, state, y, lam):
        raise NotImplementedError

    @property
    def unwrapped(self) -> AbstractFilter:
        return self._filter

    def apply(self, X, params=None, state=None, y=None, lam=None):
        return self.unwrapped.apply(X=X, params=params, state=state, y=y, lam=lam)

    def init(self, bs=None, X=None, lam=None, seed: int = 1):
        return self.unwrapped.init(bs=bs, X=X, lam=lam, seed=seed)

    def nojit(self) -> "AbstractFilterWrapper":
        self._filter = self.unwrapped.nojit()
        return self

    def train(self) -> "AbstractFilterWrapper":
        return self

    def eval(self) -> "AbstractFilterWrapper":
        return self

    def search_attr(self, attr: str):
        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            if isinstance(self.unwrapped, AbstractFilterWrapper):
                return self.unwrapped.search_attr(attr)
            else:
                return getattr(self.unwrapped, attr)

    def _pre_save(self, *args, **kwargs):
        self.unwrapped._pre_save(*args, **kwargs)

    @staticmethod
    def _post_load(
        wrapper: "AbstractFilterWrapper", *args, **kwargs
    ) -> "AbstractFilterWrapper":
        wrapper._filter = wrapper._filter._post_load(wrapper._filter, *args, **kwargs)
        return wrapper


class LPF_AbstractFilterWrapper(AbstractFilterWrapper):
    def __init__(
        self,
        filter: AbstractFilter,
        cutoff_freq: float,
        samp_freq: float,
        filtfilt: bool = True,
    ) -> None:
        super().__init__(filter)
        self._kwargs = dict(
            cutoff_freq=cutoff_freq, samp_freq=samp_freq, filtfilt=filtfilt
        )

    def apply(self, X, params=None, state=None, y=None, lam=None):
        yhat, state = super().apply(X, params, state, y, lam)
        if yhat.ndim == 4:
            yhat = jax.vmap(
                jax.vmap(
                    lambda q: x_xy.maths.quat_lowpassfilter(q, **self._kwargs),
                    in_axes=2,
                    out_axes=2,
                )
            )(yhat)
        else:
            yhat = jax.vmap(
                lambda q: x_xy.maths.quat_lowpassfilter(q, **self._kwargs),
                in_axes=1,
                out_axes=1,
            )(yhat)
        return yhat, state


class GroundTruthHeading_AbstractFilterWrapper(AbstractFilterWrapper):

    def apply(self, X, params=None, state=None, y=None, lam=None):
        yhat, state = super().apply(X, params, state, y, lam)
        if lam is None:
            lam = self.search_attr("lam")
        yhat = self.transfer_ground_truth_heading(lam, y, yhat)
        return yhat, state

    @staticmethod
    def transfer_ground_truth_heading(lam, y, yhat) -> None:
        if y is None or lam is None:
            return yhat

        yhat = jnp.array(yhat)
        for i, p in enumerate(lam):
            if p == -1:
                yhat = yhat.at[..., i, :].set(
                    x_xy.maths.quat_transfer_heading(y[..., i, :], yhat[..., i, :])
                )
        return yhat


_default_factors = dict(gyr=1 / 2.2, acc=1 / 9.81, joint_axes=1 / 0.57, dt=10.0)


class ScaleX_AbstractFilterWrapper(AbstractFilterWrapper):

    def __init__(
        self, filter: AbstractFilter, factors: dict[str, float] = _default_factors
    ) -> None:
        super().__init__(filter)
        self._factors = factors

    def apply(self, X, params=None, state=None, y=None, lam=None):
        F = X.shape[-1]
        num_batch_dims = X.ndim - 1

        if F == 6:
            X = dict(acc=X[..., :3], gyr=X[..., 3:])
        elif F == 9:
            X = dict(acc=X[..., :3], gyr=X[..., 3:6], joint_axes=X[..., 6:])
        elif F == 10:
            X = dict(
                acc=X[..., :3], gyr=X[..., 3:6], joint_axes=X[..., 6:9], dt=X[..., 9:10]
            )
        else:
            raise Exception(f"X.shape={X.shape}")
        X = {key: val * self._factors[key] for key, val in X.items()}
        X = tree_utils.batch_concat_acme(X, num_batch_dims=num_batch_dims)
        return super().apply(X, params, state, y, lam)

    def train(self) -> AbstractFilterWrapper:
        return self
        # return Train_AbstractFilterWrapper(self).train()


class Train_AbstractFilterWrapper(AbstractFilterWrapper):
    "Only required during training, afterwards can be unwrapped again"

    def init(self, bs=None, X=None, lam=None, seed: int = 1):
        return super().init(bs, self._transform(X), lam, seed)

    def apply(self, X, params=None, state=None, y=None, lam=None):
        return super().apply(self._transform(X), params, state, y, lam)

    def _transform(self, X):
        X = self._expand_dt(X)
        X = self._flatten_if_dict(X)
        return X

    # TODO investigate if really jax.Array or np.ndarray
    def _flatten_if_dict(self, X: jax.Array | dict[str, dict[str, jax.Array]]):
        if not isinstance(X, dict):
            return X
        X = X.copy()

        N = len(X)
        for i in range(N):
            assert str(i) in X

        def dict_to_tuple(d: dict[str, jax.Array]):
            tup = (d["acc"], d["gyr"])
            if "joint_axes" in d:
                tup = tup + (d["joint_axes"],)
            if "dt" in d:
                tup = tup + (d["dt"],)
            return tup

        X = [dict_to_tuple(X[str(i)]) for i in range(N)]
        X = tree_utils.tree_batch(X, backend="jax")
        X = tree_utils.batch_concat_acme(X, num_batch_dims=3).transpose((1, 2, 0, 3))
        assert X.shape[2] == N
        return X

    def _expand_dt(self, X: jax.Array | dict[str, dict[str, jax.Array]]):
        if not isinstance(X, dict) or "dt" not in X:
            return X
        X = X.copy()

        gyr = X["0"]["gyr"]
        assert gyr.ndim == 3
        T = gyr.shape[1]

        dt = X.pop("dt")
        dt = jnp.repeat(dt[:, None, :], T, axis=1)
        for seg in X:
            X[seg]["dt"] = dt
        return X

    def eval(self) -> AbstractFilterWrapper:
        return self._filter.eval()
