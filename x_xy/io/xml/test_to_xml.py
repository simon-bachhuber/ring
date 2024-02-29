import logging

import x_xy
from x_xy.base import System
from x_xy.utils import sys_compare


def test_save_sys_to_str():
    for original_sys in x_xy.io.list_load_examples():
        sys_to_xml_str = x_xy.io.save_sys_to_str(original_sys)

        logging.debug(sys_to_xml_str)

        compare_sys = x_xy.io.load_sys_from_str(sys_to_xml_str)

        assert sys_compare(
            original_sys, compare_sys
        ), f"Failed {original_sys.model_name}.xml"

        print(f"Passed {original_sys.model_name}.xml")

    def double_load_xml_to_sys(example: str) -> System:
        orig_sys = x_xy.io.load_example(example)
        exported_xml = x_xy.io.save_sys_to_str(orig_sys)
        new_sys = x_xy.io.load_sys_from_str(exported_xml)
        return new_sys

    sys_test_xml_1 = double_load_xml_to_sys("test_all_1.xml")
    sys_test_xml_2 = double_load_xml_to_sys("test_all_2.xml")

    assert not sys_compare(sys_test_xml_1, sys_test_xml_2)
