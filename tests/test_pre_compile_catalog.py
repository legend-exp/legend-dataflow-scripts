from __future__ import annotations

import os

from legenddataflowscripts.workflow import pre_compile_catalog

VALIDITY = """\
- valid_from: 20230101T000000Z
  apply:
    - det1.yaml
"""


def _make_db(path, value):
    path.mkdir(exist_ok=True)
    (path / "det1.yaml").write_text(f"val: {value}\n")
    (path / "validity.yaml").write_text(VALIDITY)


def test_pre_compile_catalog_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    dbdir = tmp_path / "statuses"
    _make_db(dbdir, 1)

    cat1 = pre_compile_catalog(dbdir)
    assert cat1.valid_for("20230201T000000Z")["val"] == 1

    cache_dir = tmp_path / "cache" / "legend-dataflow" / "precompiled-catalogs"
    cache_files = list(cache_dir.glob("statuses-*.pkl"))
    assert len(cache_files) == 1

    # second call is served from the pickle and matches the compiled result
    cat2 = pre_compile_catalog(dbdir)
    assert cat2.valid_for("20230201T000000Z") == cat1.valid_for("20230201T000000Z")

    # changing the database invalidates the cache and prunes the stale pickle
    _make_db(dbdir, 2)
    st = (dbdir / "det1.yaml").stat()
    os.utime(dbdir / "det1.yaml", (st.st_atime, st.st_mtime + 1))
    cat3 = pre_compile_catalog(dbdir)
    assert cat3.valid_for("20230201T000000Z")["val"] == 2
    new_cache_files = list(cache_dir.glob("statuses-*.pkl"))
    assert len(new_cache_files) == 1
    assert new_cache_files != cache_files
