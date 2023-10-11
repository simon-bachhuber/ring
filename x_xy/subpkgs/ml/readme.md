# RNN-based Observer (RNNO)

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
from x_xy.subpkgs import ml

xml_str = ...
sys = x_xy.load_sys_from_str(xml_str)

rnno = ml.make_rnno(sys)
```

## What the RNNO network expect as `X` and returns as `y`
Let's assume there are 6D IMUs attached to bodies `seg1` and `seg3`. Then,

```python
# X is a dict of the time-series of measurement of both outer IMUs
# e.g. 60s at 100Hz
n_timesteps = 60 * 100

X = {
    "seg1": {
        "acc": jax.Array,
        "gyr": jax.Array,
    },
    "seg2": {
        # zeros because there is no IMU on `seg2`
        "acc": jnp.zeros((n_timesteps, 3)),
        "gyr": jnp.zeros((n_timesteps, 3)),
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
## The (init, apply) function pair

```python
import tree_utils

# X must be batched
X = tree_utils.add_batch_dim(X)

rnno = ml.make_rnno(sys)
# initialize the network parameters and the initial state 
# using a random seed `key`
params, state = rnno.init(key, X)
# then we can call the network with 
yhat, _ = network.apply(params, state, X)
```
