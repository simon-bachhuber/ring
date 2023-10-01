from typing import Protocol

import x_xy


class Filter(Protocol):
    def init(self, sys: x_xy.System, X_t0: dict):
        "X_t0.shape = (features,)"
        ...

    def predict(self, X: dict) -> dict:
        "X.shape = (bs, timesteps, features)"
        ...

    def identifier(self) -> str:
        ...
