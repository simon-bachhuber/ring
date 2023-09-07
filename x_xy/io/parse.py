from .. import base
from ..scan import scan_sys


def parse_system(sys: base.System) -> base.System:
    """Initial setup of system. System object does not work unless it is parsed.
    Currently it does:
    - some consistency checks
    - populate the spatial inertia tensors
    - check that all names are unique
    - check that names are strings
    """
    assert len(sys.link_parents) == len(sys.link_types) == sys.links.batch_dim()

    for name in sys.link_names:
        assert sys.link_names.count(name) == 1, "Duplicated name in system"
        assert isinstance(name, str)

    for geom in sys.geoms:
        assert geom.link_idx in list(range(sys.num_links())) + [-1]

    inertia = _parse_system_calculate_inertia(sys)
    sys = sys.replace(links=sys.links.replace(inertia=inertia))

    return sys


def inertia_from_geometries(geometries: list[base.Geometry]) -> base.Inertia:
    inertia = base.Inertia.zero()
    for geom in geometries:
        inertia += base.Inertia.create(geom.mass, geom.transform, geom.get_it_3x3())
    return inertia


def _parse_system_calculate_inertia(sys: base.System):
    def compute_inertia_per_link(_, __, link_idx: int):
        geoms_link = []
        for geom in sys.geoms:
            if geom.link_idx == link_idx:
                geoms_link.append(geom)

        it = inertia_from_geometries(geoms_link)
        return it

    return scan_sys(sys, compute_inertia_per_link, "l", list(range(sys.num_links())))
