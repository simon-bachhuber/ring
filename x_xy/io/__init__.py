from pathlib import Path

from .omc import postprocess, process_omc
from .omc.postprocess import omc_to_xs
from .omc.utils import render_omc
from .parse import parse_system
from .xml import load_sys_from_str, load_sys_from_xml

examples_dir = Path(__file__).parent.joinpath("examples")


def load_example(name: str):
    "Load example from examples dir. `name` is without .xml extension."

    xml_path = examples_dir.joinpath(f"{name}.xml")
    return load_sys_from_xml(xml_path)


def list_examples() -> list[str]:
    import os

    def list_of_examples_in_folder(folder):
        return [ex.split(".")[0] for ex in os.listdir(folder)]

    folders = ["", "three_segs", "four_segs"]
    examples = []
    for folder in folders:
        example_folder = list_of_examples_in_folder(examples_dir.joinpath(folder))
        if len(folder) > 0:
            example_folder = [folder + "/" + ex for ex in example_folder]
        examples += example_folder

    # exclude subfolders from examples
    examples = list(set(examples) - set(folders))

    return examples
