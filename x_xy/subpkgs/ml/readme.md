# RNN-based Observer (RNNO)

## Installation

Create a new conda-env with `Python=3.10` or `Python=3.11`.
Then,
- `pip install git+https://github.com/SimiPixel/neural_networks.git`

---

## Defining a RNNO
RNNO_v2 is locally passing messages between nodes of a kinematic tree in order to estimate the pose of the kinematic tree. The order in which messages are passed is defined by a `x_xy.base.System`.

Suppose we are given a system defined by the following xml-string:

```xml
<x_xy model="dustin_exp">
    <options gravity="0 0 9.81" dt="0.01"/>
    <worldbody>
        <body name="seg1" joint="free">
            <body name="seg2" joint="ry">
                <body name="seg3" joint="rz"></body>
            </body>
        </body>
    </worldbody>
</x_xy>
```

Then, we can create a RNNO using

```python
import x_xy
from neural_networks.rnno import rnno_v2

dustin_exp_xml_str = ...
sys = x_xy.io.load_sys_from_str(dustin_exp_xml_str)

# or replace `rnno_v2` with `rnno_v1`
rnno = rnno_v2(sys)
```

## What the RNNO network expect as `X` and returns as `y`
Let's assume there are 6D IMUs attached to bodies `seg1` and `seg3`. Then,

```python
# X is a dict of the time-series of measurement of both outer IMUs
X = {
    "seg1": {
        "acc": jax.Array,
        "gyr": jax.Array,
    },
    "seg3": {
        "acc": jax.Array,
        "gyr": jax.Array,
    }
}
```
and what it returns is
```python
# y is a dict of the time-series of quaternions
# by default bodies that connect to the worldbody are skipped
y = {
    # the jax.Array should be of shape (n_timesteps, 4)
    "seg2": jax.Array,
    "seg3": jax.Array
}
```
**Note that the meaning of those returned quaternions is something you define by your training objective. If you let the training data be the relative orientation of the child body relative to its parent. Then, that's what RNNO learns and what the returned quaternions thus encode.**
## Both RNNO network is a map from

```python
network = rnno(sys)
# initialize the network parameters and the initial state 
# using a random seed `key`
params, state = network.init(key, X)
# then we can call the network with 
yhat, _ = network.apply(params, state, X)
```
where `X` has no batchsize dimension. Batching is done via `jax.vmap`.
