import logging

import ring
from ring.base import System
from ring.utils import sys_compare


def test_save_sys_to_str():
    for original_sys in ring.io.list_load_examples():
        sys_to_xml_str = ring.io.save_sys_to_str(original_sys)

        logging.debug(sys_to_xml_str)

        compare_sys = ring.io.load_sys_from_str(sys_to_xml_str)

        assert sys_compare(
            original_sys, compare_sys
        ), f"Failed {original_sys.model_name}.xml"

        print(f"Passed {original_sys.model_name}.xml")

    def double_load_xml_to_sys(example: str) -> System:
        orig_sys = ring.io.load_example(example)
        exported_xml = ring.io.save_sys_to_str(orig_sys)
        new_sys = ring.io.load_sys_from_str(exported_xml)
        return new_sys

    sys_test_xml_1 = double_load_xml_to_sys("test_all_1.xml")
    sys_test_xml_2 = double_load_xml_to_sys("test_all_2.xml")

    assert not sys_compare(sys_test_xml_1, sys_test_xml_2)
