import jax
import numpy as np
import ring


def test_jit_forward_kinematics():
    "This tests the lack of a certain bug. Details see function `_from_xml_vispy` below"
    for sys in ring.io.list_load_examples():
        for _ in range(2):
            jax.jit(ring.algorithms.forward_kinematics)(
                sys, ring.base.State.create(sys)
            )


# copied over from `abstract.py`; completely unused
ATTR = ...


def _from_xml_vispy(attr: ATTR):
    """Find all keys starting with `vispy_`, and return subdict without that prefix.
    Also convert all arrays back to list[float], because of `struct.field(False)`.
    Otherwise jitted functions with `sys` input will error on second execution, since
    it can't compare the two vispy_color arrays.
    """

    def delete_prefix(key):
        len_suffix = len(key.split("_")[0]) + 1
        return key[len_suffix:]

    dict_no_prefix = {
        delete_prefix(k): attr[k] for k in attr if k.split("_")[0] == "vispy"
    }

    # convert arrays -> list[float]
    to_list = lambda ele: (
        ele.tolist() if isinstance(ele, (np.ndarray, jax.Array)) else ele
    )
    return {key: to_list(value) for key, value in dict_no_prefix.items()}
