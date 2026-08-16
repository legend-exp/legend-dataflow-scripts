from __future__ import annotations

import json
from pathlib import Path

import h5py


def convert_parents_to_structs(h5group):
    """Annotate HDF5 parent groups with LGDO ``struct`` datatype attributes.

    When a new alias (hard link) is created inside an HDF5 file the parent
    groups may not carry the ``datatype`` attribute expected by the LGDO reader.
    This function walks up the HDF5 group hierarchy from *h5group* and ensures
    every ancestor group carries a ``datatype`` attribute of the form
    ``struct{child1,child2,…}``.

    The root group is only updated if it already carries a ``datatype``; the
    walk always stops there.

    Parameters
    ----------
    h5group : h5py.Group
        Leaf group whose parent hierarchy should be annotated.
    """
    # the walk must be iterative: in h5py the root group is its own parent, so
    # recursing on ``h5group.parent`` never terminates once ``/`` is reached
    while h5group.name != "/":
        parent = h5group.parent
        child = h5group.name.rsplit("/", 1)[-1]

        if "datatype" not in parent.attrs:
            # never invent a datatype on the file root
            if parent.name == "/":
                return
            parent.attrs.update({"datatype": "struct{" + child + "}"})
        else:
            datatype = parent.attrs["datatype"]
            if not (datatype.startswith("struct{") and datatype.endswith("}")):
                msg = (
                    f"cannot register alias {child!r}: parent group "
                    f"{parent.name!r} has non-struct datatype {datatype!r}"
                )
                raise ValueError(msg)
            # membership must be tested against the parsed field list, not by
            # substring (e.g. 'raw' is a substring of 'raw_blind')
            fields = [
                field.strip()
                for field in datatype[len("struct{") : -1].split(",")
                if field.strip()
            ]
            if child in fields:
                # ancestors were annotated when this member was first added
                return
            parent.attrs.update(
                {"datatype": "struct{" + ",".join([*fields, child]) + "}"}
            )

        h5group = parent


def alias_table(file: str | Path, mapping: str):
    """Create HDF5 hard-link aliases for existing tables in an LH5 file.

    Given a mapping of ``{source_path: alias_path}`` pairs, this function
    opens *file* in append mode and creates one HDF5 hard link per entry so
    that the data can be accessed under both the original and the alias path.
    If *alias_path* is a list or tuple each element is registered as a
    separate alias.  After each alias is created, parent groups are annotated
    with LGDO ``struct`` datatype metadata via
    :func:`convert_parents_to_structs`.

    The function can also accept a JSON-encoded string or a list of mappings
    (which are applied sequentially).

    Parameters
    ----------
    file : str or pathlib.Path
        Path to the LH5 (HDF5) file to modify.
    mapping : str or dict or list
        One of:

        * A JSON string that decodes to a ``dict`` or ``list``.
        * A ``dict`` mapping source paths to alias path(s).
        * A ``list`` of such dicts, applied recursively.
    """
    if isinstance(mapping, str):
        mapping = json.loads(mapping)
    if isinstance(mapping, list):
        for m in mapping:
            alias_table(file, m)
        return
    with h5py.File(file, "a") as f:
        for raw_id, alias in mapping.items():
            if raw_id in f:
                if isinstance(alias, list | tuple):
                    for a in alias:
                        f[a] = f[raw_id]
                        convert_parents_to_structs(f[a])
                else:
                    f[alias] = f[raw_id]
                    convert_parents_to_structs(f[alias])
