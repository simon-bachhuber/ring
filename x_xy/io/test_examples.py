import x_xy


def test_examples():
    for example in x_xy.io.list_examples():
        x_xy.io.load_example(example)
