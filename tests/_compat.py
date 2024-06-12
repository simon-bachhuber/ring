from diodem import benchmark
import jax


def unbatch_gen(gen):
    def _gen(*args):
        return jax.tree_map(lambda arr: arr[0], gen(*args))

    return _gen


def _load_sys(exp_id: int):
    return benchmark._benchmark._load_sys(exp_id)
