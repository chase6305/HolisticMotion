"""Validated scalar joint directions at the Python model import boundary."""

import math
from xml.dom import Node, minidom
from xml.parsers.expat import ExpatError


def normalized_urdf_xml(path):
    """Read an explicit URDF without modifying it or resolving mesh resources.

    Use a document-local DOM so default XML namespaces and unrelated elements
    keep their names when the normalized model is passed to Pinocchio.
    """
    try:
        document = minidom.parse(str(path))
    except ExpatError as error:
        raise ValueError(f"Unable to parse URDF model: {path}") from error
    with document:
        for joint in document.documentElement.childNodes:
            if (
                joint.nodeType != Node.ELEMENT_NODE
                or joint.tagName != "joint"
                or joint.getAttribute("type")
                not in ("revolute", "continuous", "prismatic")
            ):
                continue
            axis = next(
                (
                    child
                    for child in joint.childNodes
                    if child.nodeType == Node.ELEMENT_NODE and child.tagName == "axis"
                ),
                None,
            )
            if axis is None:
                # URDF defaults a missing scalar-joint axis to the X direction.
                continue
            message = (
                "URDF joint axis must contain three finite values and be nonzero: "
                + joint.getAttribute("name")
            )
            try:
                values = tuple(
                    float(value) for value in axis.getAttribute("xyz").split()
                )
            except ValueError as error:
                raise ValueError(message) from error
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError(message)
            scale = max(abs(value) for value in values)
            if scale == 0.0:
                raise ValueError(message)
            values = tuple(value / scale for value in values)
            norm = math.sqrt(sum(value * value for value in values))
            axis.setAttribute(
                "xyz", " ".join(format(value / norm, ".17g") for value in values)
            )
        return document.toxml()
