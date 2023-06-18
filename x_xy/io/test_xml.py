import x_xy
from pathlib import Path


def test_xml_parsing():
    for example_name in x_xy.io.list_examples():
        xml_path = Path(__file__).parent.joinpath(f"examples/{example_name}.xml")
        original_sys: x_xy.System = x_xy.io.xml.load_sys_from_xml(xml_path)

        sys_to_xml_str = x_xy.io.xml.system_to_xml_str(original_sys)
        compare_sys = x_xy.io.xml.load_sys_from_str(sys_to_xml_str)

        assert original_sys == compare_sys, (f"Failed to parse {example_name}.xml \
                    \n Original: {original_sys} \
                    \n Parsed: {compare_sys}")

        print(f"Passed {example_name}.xml")


if __name__ == "__main__":
    test_xml_parsing()
