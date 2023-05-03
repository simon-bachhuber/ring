from x_xy import base, scan


def parse_system(sys: base.System) -> base.System:
    """Initial setup of system. System object does not work unless it is parsed.
    Currently it does:
    - some consistency checks
    - populate the spatial inertia tensors
    - check that all names are unique
    - check that names are strings
    """
    assert (
        len(sys.link_parents)
        == len(sys.link_types)
        == len(sys.geoms)
        == sys.links.batch_dim()
    )

    for name in sys.link_names:
        assert sys.link_names.count(name) == 1, "Duplicated name in system"
        assert isinstance(name, str)

    for geometries_links in sys.geoms:
        assert isinstance(geometries_links, list)

    inertia = _parse_system_calculate_inertia(sys)
    sys = sys.replace(links=sys.links.replace(inertia=inertia))

    return sys


def inertia_from_geometries(geometries: list[base.Geometry]) -> base.Inertia:
    inertia = base.Inertia.zero()
    for geom in geometries:
        inertia += base.Inertia.create(geom.mass, geom.transform, geom.get_it_3x3())
    return inertia


def _parse_system_calculate_inertia(sys: base.System):
    def compute_inertia_per_link(_, __, geometries_link):
        it = inertia_from_geometries(geometries_link)
        return it

    return scan.tree(sys, compute_inertia_per_link, "l", sys.geoms)
