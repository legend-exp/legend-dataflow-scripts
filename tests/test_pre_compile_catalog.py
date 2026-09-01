from __future__ import annotations

import os
import pickle
from unittest.mock import patch

from legenddataflowscripts.workflow import pre_compile_catalog

VALIDITY = """\
- valid_from: 20230101T000000Z
  apply:
    - det1.yaml
"""


def _make_db(path, value):
    path.mkdir(parents=True, exist_ok=True)
    (path / "det1.yaml").write_text(f"val: {value}\n")
    (path / "validity.yaml").write_text(VALIDITY)


def test_pre_compile_catalog_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    dbdir = tmp_path / "statuses"
    _make_db(dbdir, 1)

    cat1 = pre_compile_catalog(dbdir)
    assert cat1.valid_for("20230201T000000Z")["val"] == 1

    cache_root = tmp_path / "cache" / "legend-dataflow" / "precompiled-catalogs"
    cache_files = list(cache_root.glob("statuses-*/*.pkl"))
    assert len(cache_files) == 1

    # the second call must be served from the pickle: fail if it recompiles
    with patch(
        "legenddataflowscripts.workflow.pre_compile_catalog.Catalog.read_from",
        side_effect=AssertionError("cache miss: catalog was recompiled"),
    ):
        cat2 = pre_compile_catalog(dbdir)
    assert cat2.valid_for("20230201T000000Z") == cat1.valid_for("20230201T000000Z")

    # changing the database invalidates the cache and prunes the stale pickle
    _make_db(dbdir, 2)
    st = (dbdir / "det1.yaml").stat()
    os.utime(dbdir / "det1.yaml", ns=(st.st_atime_ns, st.st_mtime_ns + 1))
    cat3 = pre_compile_catalog(dbdir)
    assert cat3.valid_for("20230201T000000Z")["val"] == 2
    new_cache_files = list(cache_root.glob("statuses-*/*.pkl"))
    assert len(new_cache_files) == 1
    assert new_cache_files != cache_files


def test_pre_compile_catalog_same_basename_dirs(tmp_path, monkeypatch):
    """Same-named directories in different locations get separate cache
    namespaces, so rebuilding one must not evict the other's pickle."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    db_a = tmp_path / "a" / "statuses"
    db_b = tmp_path / "b" / "statuses"
    _make_db(db_a, 1)
    _make_db(db_b, 2)

    cat_a = pre_compile_catalog(db_a)
    cat_b = pre_compile_catalog(db_b)
    assert cat_a.valid_for("20230201T000000Z")["val"] == 1
    assert cat_b.valid_for("20230201T000000Z")["val"] == 2

    # both caches survive, and each directory still hits its own pickle
    cache_root = tmp_path / "cache" / "legend-dataflow" / "precompiled-catalogs"
    assert len(list(cache_root.glob("statuses-*/*.pkl"))) == 2
    with patch(
        "legenddataflowscripts.workflow.pre_compile_catalog.Catalog.read_from",
        side_effect=AssertionError("cache miss: catalog was recompiled"),
    ):
        assert pre_compile_catalog(db_a).valid_for("20230201T000000Z")["val"] == 1
        assert pre_compile_catalog(db_b).valid_for("20230201T000000Z")["val"] == 2


def test_pre_compile_catalog_rejects_bad_pickle(tmp_path, monkeypatch):
    """A corrupt or wrong-typed cache file is treated as a miss."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    dbdir = tmp_path / "statuses"
    _make_db(dbdir, 1)

    cat1 = pre_compile_catalog(dbdir)
    cache_root = tmp_path / "cache" / "legend-dataflow" / "precompiled-catalogs"
    (cache_file,) = cache_root.glob("statuses-*/*.pkl")

    cache_file.write_bytes(b"not a pickle")
    assert pre_compile_catalog(dbdir).valid_for("20230201T000000Z") == cat1.valid_for(
        "20230201T000000Z"
    )

    cache_file.write_bytes(pickle.dumps({"not": "a catalog"}))
    assert pre_compile_catalog(dbdir).valid_for("20230201T000000Z") == cat1.valid_for(
        "20230201T000000Z"
    )
