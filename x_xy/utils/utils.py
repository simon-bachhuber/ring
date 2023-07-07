import jax
import jax.numpy as jnp

from x_xy.base import _Base

JIT_WARN = True


def disable_jit_warn():
    global JIT_WARN
    JIT_WARN = False


def tree_equal(a, b):
    "Copied from Marcel / Thomas"
    if type(a) is not type(b):
        return False
    if isinstance(a, _Base):
        return tree_equal(a.__dict__, b.__dict__)
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(tree_equal(a[k], b[k]) for k in a.keys())
    if isinstance(a, (tuple, list)):
        if len(a) != len(b):
            return False
        return all(tree_equal(a[i], b[i]) for i in range(len(a)))
    if isinstance(a, jax.Array):
        return jnp.array_equal(a, b)
    return a == b


def sys_compare(sys1, sys2, verbose: bool = True) -> bool:
    d1 = sys1.__dict__
    d2 = sys2.__dict__
    for key in d1:
        if not tree_equal(d1[key], d2[key]):
            if verbose:
                print(f"Systems different in attribute `.{key}`")
                print(f"{d1[key]} NOT EQUAL {d2[key]}")
            return False
    return True
