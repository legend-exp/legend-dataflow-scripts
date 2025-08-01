from __future__ import annotations

import json
from pathlib import Path

import h5py


def convert_parents_to_structs(h5group):
    if h5group.parent.name != "/" and len(h5group.parent.attrs) == 0:
        h5group.parent.attrs.update(
            {"datatype": "struct{" + h5group.name.split("/")[-1] + "}"}
        )
    elif (
        len(h5group.parent.attrs) > 0
        and h5group.name.split("/")[-1] not in h5group.parent.attrs["datatype"]
    ):
        h5group.parent.attrs.update(
            {
                "datatype": h5group.parent.attrs["datatype"][:-1]
                + ","
                + h5group.name.split("/")[-1]
                + "}"
            }
        )
    else:
        return
    convert_parents_to_structs(h5group.parent)
    return


def alias_table(file: str | Path, mapping: str):
    """
    Create an alias table for the given file and mapping.

    Args:
        file (str): Path to the input file.
        mapping (dict): Mapping of current table name and alias table name.

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
