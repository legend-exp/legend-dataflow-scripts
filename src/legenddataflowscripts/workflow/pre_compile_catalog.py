from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from dbetto import TextDB
from dbetto.catalog import Catalog


def _dir_state_digest(validity_path: Path) -> str:
    """Digest of the directory contents (paths, mtimes, sizes) plus the dbetto
    version, identifying one compiled state of the catalog."""
    state = [version("dbetto"), str(validity_path.resolve())]
    for f in sorted(validity_path.rglob("*")):
        if f.is_file():
            st = f.stat()
            state.append(f"{f.relative_to(validity_path)}:{st.st_mtime}:{st.st_size}")
    return hashlib.sha256("\n".join(state).encode()).hexdigest()[:16]


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "legend-dataflow" / "precompiled-catalogs"


def pre_compile_catalog(validity_path: str | Path):
    """Pre-compile a dbetto validity catalog for fast repeated access.

    Reads the ``validity.yaml`` catalog from *validity_path* and, for each
    system and each entry in the catalog, eagerly loads the corresponding
    :class:`dbetto.TextDB` state (instead of loading it lazily on first
    access).  The resulting :class:`dbetto.catalog.Catalog` can be serialised
    and reused across many Snakemake jobs without re-parsing YAML on every
    invocation.

    The compiled catalog is cached on disk under
    ``$XDG_CACHE_HOME/legend-dataflow/precompiled-catalogs``, keyed on the
    directory contents (file mtimes/sizes) and the dbetto version, so repeated
    workflow parses skip the eager compilation entirely.

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

    digest = _dir_state_digest(validity_path)
    cache_file = _cache_dir() / f"{validity_path.name}-{digest}.pkl"
    if cache_file.is_file():
        try:
            with cache_file.open("rb") as f:
                return pickle.load(f)
        except Exception:  # corrupt/stale pickle: rebuild below
            pass

    catalog = Catalog.read_from(validity_path / "validity.yaml")
    entries = {}
    textdb = TextDB(validity_path, lazy=False)
    for system in catalog.entries:
        entries[system] = []
        for entry in catalog.entries[system]:
            db = textdb.on(
                datetime.fromtimestamp(entry.valid_from, tz=UTC), category=system
            )
            new_entry = Catalog.Entry(entry.valid_from, db)
            entries[system].append(new_entry)
    compiled = Catalog(entries)

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        for stale in cache_file.parent.glob(f"{validity_path.name}-*.pkl"):
            stale.unlink(missing_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=cache_file.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(compiled, f, protocol=pickle.HIGHEST_PROTOCOL)
            Path(tmp_name).replace(cache_file)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
    except Exception:
        pass  # caching is best-effort; a failed write must not break parsing

    return compiled
