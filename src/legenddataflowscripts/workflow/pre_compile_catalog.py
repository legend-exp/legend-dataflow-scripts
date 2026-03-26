from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from dbetto import TextDB
from dbetto.catalog import Catalog


def pre_compile_catalog(validity_path: str | Path):
    """Pre-compile a dbetto validity catalog for fast repeated access.

    Reads the ``validity.yaml`` catalog from *validity_path* and, for each
    system and each entry in the catalog, eagerly loads the corresponding
    :class:`dbetto.TextDB` state (instead of loading it lazily on first
    access).  The resulting :class:`dbetto.catalog.Catalog` can be serialised
    and reused across many Snakemake jobs without re-parsing YAML on every
    invocation.

    Parameters
    ----------
    validity_path : str or pathlib.Path
        Directory containing the ``validity.yaml`` file and all referenced
        database files.

    Returns
    -------
    dbetto.catalog.Catalog
        Pre-compiled catalog with all entries eagerly resolved.
    """
    if isinstance(validity_path, str):
        validity_path = Path(validity_path)
    catalog = Catalog.read_from(validity_path / "validity.yaml")
    entries = {}
    textdb = TextDB(validity_path, lazy=False)
    for system in catalog.entries:
        entries[system] = []
        for entry in catalog.entries[system]:
            db = textdb.on(
                datetime.fromtimestamp(entry.valid_from, tz=UTC), system=system
            )
            new_entry = Catalog.Entry(entry.valid_from, db)
            entries[system].append(new_entry)
    return Catalog(entries)
