import ring


def test_examples():
    for example in ring.io.list_examples():
        ring.io.load_example(example)
