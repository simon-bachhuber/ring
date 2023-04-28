from .from_xml import load_sys_from_str, load_sys_from_xml
from .parse_system import parse_system


def load_example(name: str):
    "Load example from examples dir. `name` is without .xml extension."
    from pathlib import Path

    xml_path = Path(__file__).parent.joinpath(f"examples/{name}.xml")
    return load_sys_from_xml(xml_path)


def list_examples() -> list[str]:
    import os
    from pathlib import Path

    examples_dir = Path(__file__).parent.joinpath("examples")

    return [ex.split(".")[0] for ex in os.listdir(examples_dir)]
