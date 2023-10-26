from pathlib import Path

import pytest

from x_xy.utils import delete_download_cache
from x_xy.utils import dict_to_nested
from x_xy.utils import dict_union
from x_xy.utils import download_from_repo
from x_xy.utils import tree_equal


def test_dict_union():
    d1 = dict(seg1=dict(inp1=1.0, inp3=1.0), seg3=dict(inp1=1.0))
    d2 = dict(seg1=dict(inp1=2.0), seg2=dict(inp2=3.0))
    d3 = dict(seg1=dict(inp2=2.0), seg2=dict(inp2=3.0))

    with pytest.raises(AssertionError):
        dict_union(d1, d2)

    assert tree_equal(
        dict(seg1=dict(inp1=2.0, inp3=1.0), seg2=dict(inp2=3.0), seg3=dict(inp1=1.0)),
        dict_union(d1, d2, overwrite=True),
    )

    assert tree_equal(
        dict(
            seg1=dict(inp1=1.0, inp2=2.0, inp3=1.0),
            seg2=dict(inp2=3.0),
            seg3=dict(inp1=1.0),
        ),
        dict_union(d1, d3, overwrite=False),
    )


def test_dict_nest():
    d = dict(x=1)
    assert tree_equal(dict(x=dict(fancy=1)), dict_to_nested(d, "fancy"))


def test_download_from_repo():
    delete_download_cache(only="docs/img")
    icon_svg = "docs/img/icon.svg"
    path_on_disk = Path(download_from_repo(icon_svg, "x_xy_v2"))

    assert path_on_disk.exists() and path_on_disk.is_file()

    delete_download_cache(only="params/ry_rz")
    params = "params/ry_rz/params_ry_rz.pickle"
    path_on_disk = Path(download_from_repo(params))

    assert path_on_disk.exists() and path_on_disk.is_file()
