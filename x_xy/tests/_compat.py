import jax


def unbatch_gen(gen):
    def _gen(*args):
        return jax.tree_map(lambda arr: arr[0], gen(*args))

    return _gen
