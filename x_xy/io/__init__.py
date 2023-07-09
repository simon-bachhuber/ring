from pathlib import Path

from x_xy.utils import parse_path

from .omc import postprocess, process_omc
from .omc.postprocess import omc_to_xs
from .omc.utils import render_omc
from .parse import parse_system
from .xml import load_sys_from_str, load_sys_from_xml

EXAMPLES_DIR = Path(__file__).parent.joinpath("examples")
FOLDERS = ["", "test_morph_system"]
EXCLUDE_FOLDERS = ["DELETE", "experiments"]


def load_example(name: str):
    "Load example from examples dir."

    xml_path = parse_path(EXAMPLES_DIR, name, extension="xml")
    return load_sys_from_xml(xml_path)


def load_experiment(name: str):
    "Load experiment from examples/experiment dir."

    xml_path = parse_path(EXAMPLES_DIR, "experiment", name, extension="xml")
    return load_sys_from_xml(xml_path)


def list_examples() -> list[str]:
    import os

    def list_of_examples_in_folder(folder):
        return [ex.split(".")[0] for ex in os.listdir(folder)]

    examples = []
    for folder in FOLDERS:
        example_folder = list_of_examples_in_folder(EXAMPLES_DIR.joinpath(folder))
        if len(folder) > 0:
            example_folder = [folder + "/" + ex for ex in example_folder]
        examples += example_folder

    # exclude subfolders from examples
    examples = list(set(examples) - set(FOLDERS) - set(EXCLUDE_FOLDERS))

    return examples
