from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from typing import IO
from xml.dom import minidom

import landlab
import meshio
import numpy as np


def vtu_dump(
    grid: landlab.ModelGrid,
    stream: IO[str] | None = None,
    include: str = "*",
    exclude: Sequence[str] | None = None,
    z_coord: float = 0.0,
    at: str = "node",
) -> str | None:
    """Return a VTU representation of a grid.

    Parameters
    ----------
    grid : landlab.ModelGrid
        Grid containing fields to output.
    stream : file-like object or None, optional
        Stream to write VTU data to. If ``None`` the string is returned.
    include : sequence of str or "*", optional
        Fields to include.
    exclude : sequence of str or None, optional
        Fields to exclude.
    z_coord : float, optional
        Elevation to use for 2D grids.
    at : {"node", "cell"}, optional
        Location of values within the grid.

    Returns
    -------
    str or None
        VTU representation or ``None`` if written to ``stream``.
    """
    mask = grid.status_at_node == grid.BC_NODE_IS_CLOSED
    saved_fields = {
        name: grid.at_node[name]
        for name in grid.at_node
        if np.issubdtype(grid.at_node[name].dtype, np.floating)
    }
    for name, array in saved_fields.items():
        grid.at_node[name] = array.copy()
        grid.at_node[name][mask] = np.nan

    with tempfile.NamedTemporaryFile(suffix=".vtk", mode="w+", delete=False) as tmp:
        tmp.write(
            landlab.io.legacy_vtk.dump(
                grid, include=include, exclude=exclude, z_coord=z_coord, at=at
            )
        )
        tmp.flush()
        mesh = meshio.read(tmp.name)

    for name, array in saved_fields.items():
        grid.at_node[name] = array

    with tempfile.NamedTemporaryFile(suffix=".vtu", mode="r+", delete=False) as tmp:
        tmp.close()
        meshio.write(tmp.name, mesh)
        with open(tmp.name, encoding="utf-8") as f:
            contents = f.read()

    content = "\n".join(
        [
            line
            for line in minidom.parseString(contents)
            .toprettyxml(indent="  ")
            .splitlines()
            if line.strip()
        ]
    )

    if stream is None:
        return content
    else:
        stream.write(content)
        return None


def pvtu_dump(grid: landlab.ModelGrid, vtu_files: Sequence[str] = ()) -> str:
    """Return a PVTU file referencing multiple VTU files.

    Parameters
    ----------
    grid : landlab.ModelGrid
        Grid from which point fields are derived.
    vtu_files : sequence of str, optional
        Paths to VTU files that comprise the full grid.

    Returns
    -------
    str
        XML string describing the parallel unstructured grid.
    """
    vtkfile = ET.Element(
        "VTKFile", type="PUnstructuredGrid", version="1.0", byte_order="LittleEndian"
    )
    pgrid = ET.SubElement(vtkfile, "PUnstructuredGrid", GhostLevel="0")

    ppoints = ET.SubElement(pgrid, "PPoints")
    ET.SubElement(
        ppoints, "PDataArray", type="Float32", NumberOfComponents="3", Name="Points"
    )

    if grid.at_node:
        pdata = ET.SubElement(pgrid, "PPointData")
        for name in grid.at_node:
            ET.SubElement(
                pdata, "PDataArray", type="Float32", Name=name, NumberOfComponents="1"
            )

    for f in vtu_files:
        ET.SubElement(pgrid, "Piece", Source=os.path.basename(f))

    tree = ET.ElementTree(vtkfile)

    root = tree.getroot()
    if root is None:
        raise RuntimeError("tree has no root")
    parsed = minidom.parseString(ET.tostring(root))

    return parsed.toprettyxml(indent="  ")
