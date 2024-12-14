import warnings
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement
from xml.etree.ElementTree import tostring

import jax.numpy as jnp
from tree_utils import batch_concat

from ring import base

from . import abstract
from .abstract import _to_str


def save_sys_to_str(sys: base.System, warn: bool = True) -> str:
    for joint_type in sys.links.joint_params:
        for i, link_name in enumerate(sys.link_names):
            joint_params_flat = batch_concat((sys.links[i]).joint_params[joint_type], 0)
            if warn and (not jnp.all(joint_params_flat == 0.0)):
                warnings.warn(
                    "The system has `sys.links.joint_params` unequal to the 'default'"
                    f" value (of zeros). In particular the link `{link_name}` has for"
                    f" the jointtype `{joint_type}` the values {joint_params_flat}. "
                    "This will not be preserved in the xml."
                )
    global_index_map = {qd: sys.idx_map(qd) for qd in ["q", "d"]}

    # Create root element
    x_xy = Element("x_xy")
    x_xy.set("model", sys.model_name)

    options = SubElement(x_xy, "options")
    options.set("dt", str(sys.dt))
    options.set("gravity", _to_str(sys.gravity))

    # Create worldbody
    worldbody = SubElement(x_xy, "worldbody")

    def process_link(link_idx: int, parent_elem: Element):
        link = sys.links[link_idx]
        link_typ = sys.link_types[link_idx]
        link_name = sys.link_names[link_idx]

        # Create body element
        body = SubElement(parent_elem, "body")
        body.set("joint", link_typ)
        body.set("name", link_name)

        # Set attributes
        abstract.AbsTrans.to_xml(body, link.transform1)
        abstract.AbsPosMinMax.to_xml(body, link.pos_min, link.pos_max)
        abstract.AbsDampArmaStiffZero.to_xml(
            body,
            sys.link_damping[global_index_map["d"][link_name]],
            sys.link_armature[global_index_map["d"][link_name]],
            sys.link_spring_stiffness[global_index_map["d"][link_name]],
            sys.link_spring_zeropoint[global_index_map["q"][link_name]],
            base.Q_WIDTHS[link_typ],
            base.QD_WIDTHS[link_typ],
            link_typ,
        )

        # Add geometry elements
        geoms = sys.geoms
        for geom in geoms:
            if geom.link_idx == link_idx:
                geom_elem = SubElement(body, "geom")
                abstract_class = abstract.geometry_to_abstract[type(geom)]
                abstract_class.to_xml(geom_elem, geom)

        # Maybe add omc element
        omc_link = sys.omc[link_idx]
        if omc_link is not None:
            omc_elem = SubElement(body, "omc")
            abstract.AbsMaxCoordOMC.to_xml(omc_elem, omc_link)

        # Recursively process child links
        for child_idx, parent_idx in enumerate(sys.link_parents):
            if parent_idx == link_idx:
                process_link(child_idx, body)

    for root_link_idx, parent_idx in enumerate(sys.link_parents):
        if parent_idx == -1:
            process_link(root_link_idx, worldbody)

    # Pretty print xml
    xml_str = parseString(tostring(x_xy)).toprettyxml(indent="  ")
    return xml_str


def save_sys_to_xml(sys: base.System, xml_path: str) -> None:
    xml_str = save_sys_to_str(sys)
    with open(xml_path, "w") as f:
        f.write(xml_str)
