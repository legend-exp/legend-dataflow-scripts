from __future__ import annotations

import json
from pathlib import Path

import h5py


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
    with h5py.File(file, "a") as f:
        for raw_id, alias in mapping.items():
            if raw_id in f:
                if isinstance(alias, list | tuple):
                    for a in alias:
                        f[a] = f[raw_id]
                else:
                    f[alias] = f[raw_id]
