from importlib import import_module as _import_module
import io
import pickle
import random
from typing import Optional
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import tree_utils

from ring.base import _Base
from ring.base import Geometry

from .path import parse_path


def tree_equal(a, b):
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
    if isinstance(a, (jax.Array, np.ndarray)):
        return jnp.allclose(a, b)
    return a == b


def _sys_compare_unsafe(sys1, sys2, verbose: bool, prefix: str) -> bool:
    d1 = sys1.__dict__
    d2 = sys2.__dict__
    for key in d1:
        if isinstance(d1[key], _Base):
            if not _sys_compare_unsafe(d1[key], d2[key], verbose, prefix + "." + key):
                return False
        elif isinstance(d1[key], list) and isinstance(d1[key][0], Geometry):
            for ele1, ele2 in zip(d1[key], d2[key]):
                if not _sys_compare_unsafe(ele1, ele2, verbose, prefix + "." + key):
                    return False
        else:
            if not tree_equal(d1[key], d2[key]):
                if verbose:
                    print(f"Systems different in attribute `sys{prefix}.{key}`")
                    print(f"{repr(d1[key])} NOT EQUAL {repr(d2[key])}")
                return False
    return True


def sys_compare(sys1, sys2, verbose: bool = True):
    equalA = _sys_compare_unsafe(sys1, sys2, verbose, "")
    equalB = tree_equal(sys1, sys2)
    assert equalA == equalB
    return equalA


def to_list(obj: object) -> list:
    "obj -> [obj], if it isn't already a list."
    if not isinstance(obj, list):
        return [obj]
    return obj


def dict_union(
    d1: dict[str, jax.Array] | dict[str, dict[str, jax.Array]],
    d2: dict[str, jax.Array] | dict[str, dict[str, jax.Array]],
    overwrite: bool = False,
) -> dict:
    "Builds the union between two nested dictonaries."
    # safety copying; otherwise this function would mutate out of scope
    d1 = pytree_deepcopy(d1)
    d2 = pytree_deepcopy(d2)

    for key2 in d2:
        if key2 not in d1:
            d1[key2] = d2[key2]
        else:
            if not isinstance(d2[key2], dict) or not isinstance(d1[key2], dict):
                raise Exception(f"d1.keys()={d1.keys()}; d2.keys()={d2.keys()}")

            for key_nested in d2[key2]:
                if not overwrite:
                    assert (
                        key_nested not in d1[key2]
                    ), f"d1.keys()={d1[key2].keys()}; d2.keys()={d2[key2].keys()}"

            d1[key2].update(d2[key2])
    return d1


def dict_to_nested(
    d: dict[str, jax.Array], add_key: str
) -> dict[str, dict[str, jax.Array]]:
    "Nests a dictonary by inserting a single key dictonary."
    return {key: {add_key: d[key]} for key in d.keys()}


def save_figure_to_rgba(fig) -> np.ndarray:
    with io.BytesIO() as buff:
        fig.savefig(buff, format="raw")
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im


def pytree_deepcopy(tree):
    "Recursivley copies a pytree."
    if isinstance(tree, (int, float, jax.Array)):
        return tree
    elif isinstance(tree, np.ndarray) or np.isscalar(tree):
        return tree.copy()
    elif isinstance(tree, list):
        return [pytree_deepcopy(ele) for ele in tree]
    elif isinstance(tree, tuple):
        return tuple(pytree_deepcopy(ele) for ele in tree)
    elif isinstance(tree, dict):
        return {key: pytree_deepcopy(value) for key, value in tree.items()}
    elif isinstance(tree, _Base):
        return tree
    else:
        raise NotImplementedError(f"Not implemented for type={type(tree)}")


def import_lib(
    lib: str,
    required_for: Optional[str] = None,
    lib_pypi: Optional[str] = None,
):
    try:
        return _import_module(lib)
    except ImportError:
        _required = ""
        if required_for is not None:
            _required = f" but it is required for {required_for}"
        if lib_pypi is None:
            lib_pypi = lib
        error_msg = (
            f"Could not import `{lib}`{_required}. "
            f"Please install with `pip install {lib_pypi}`"
        )
        raise ImportError(error_msg)


def pickle_save(obj, path, overwrite: bool = False):
    path = parse_path(path, extension="pickle", file_exists_ok=overwrite)
    with open(path, "wb") as file:
        pickle.dump(obj, file, protocol=5)


def pickle_load(
    path,
):
    path = parse_path(path, extension="pickle", require_is_file=True)
    with open(path, "rb") as file:
        obj = pickle.load(file)
    return obj


def primes(n: int) -> list[int]:
    "Primefactor decomposition in ascending order."
    primfac = []
    d = 2
    while d * d <= n:
        while (n % d) == 0:
            primfac.append(d)  # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
        primfac.append(n)
    return primfac


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def replace_elements_w_nans(
    list_of_data: list[tree_utils.PyTree],
    include_elements: Optional[list[int]] = None,
    verbose: bool = False,
) -> list[tree_utils.PyTree]:
    if include_elements is None:
        include_elements = list(range(len(list_of_data)))

    assert min(include_elements) >= 0
    assert max(include_elements) < len(list_of_data)

    def _is_nan(ele: tree_utils.PyTree, i: int, verbose: bool):
        isnan = np.any(
            [np.any(np.isnan(arr)) for arr in jax.tree_util.tree_leaves(ele)]
        )
        if isnan:
            if verbose:
                print(f"Sample with idx={i} is nan. It will be replaced.")
            return True
        return False

    list_of_isnan = [int(_is_nan(e, 0, False)) for e in list_of_data]
    perc_of_isnan = sum(list_of_isnan) / len(list_of_data)

    if perc_of_isnan >= 0.02:
        warnings.warn(
            f"{perc_of_isnan * 100}% of {len(list_of_data)} datapoints are NaN"
        )
        assert perc_of_isnan != 1

    list_of_data_nonan = []
    for i, ele in enumerate(list_of_data):
        if _is_nan(ele, i, verbose):
            while True:
                j = random.choice(include_elements)
                ele_j = list_of_data[j]
                if not _is_nan(ele_j, j, verbose):
                    ele = pytree_deepcopy(ele_j)
                    break
        list_of_data_nonan.append(ele)
    return list_of_data_nonan
