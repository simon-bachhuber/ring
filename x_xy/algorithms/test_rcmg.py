import jax

import x_xy


def test_rcmg():
    for example in x_xy.io.list_examples():
        sys = x_xy.io.load_example(example)
        config = x_xy.algorithms.RCMG_Config(T=1.0)
        generator = x_xy.algorithms.build_generator(sys, config)
        bs = 8
        generator = x_xy.algorithms.batch_generator(generator, bs)

        x_xy.utils.disable_jit_warn()
        seed = jax.random.PRNGKey(
            1,
        )
        qs, xs = generator(seed)

        assert qs.shape == (bs, 100, sys.q_size())
