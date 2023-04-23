from .from_xml import load_sys_from_str, load_sys_from_xml
from .parse_system import parse_system


def load_example(name: str):
    from pathlib import Path

    xml_path = Path(__file__).parent.joinpath(f"examples/{name}.xml")
    return load_sys_from_xml(xml_path)
