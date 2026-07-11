"""Tests for the shared helpers in ``legenddataflowscripts.utils``."""

from __future__ import annotations

import logging
import sys

import h5py
import numpy as np
import pytest
import yaml

from legenddataflowscripts.utils import (
    alias_table,
    build_log,
    check_pulser_mask,
    expand_filelist,
    get_channel_config,
    get_pulser_mask,
    get_rule_config,
    require_config_keys,
)


def test_get_channel_config():
    mapping = {"ch1000000": {"a": 1}, "__default__": {"a": 0}}
    assert get_channel_config(mapping, "ch1000000") == {"a": 1}
    assert get_channel_config(mapping, "V99999A") == {"a": 0}

    # channel present but no __default__ must not raise
    assert get_channel_config({"ch1000000": {"a": 1}}, "ch1000000") == {"a": 1}

    with pytest.raises(KeyError, match="channel V99999A has no entry in qc_config"):
        get_channel_config({"ch1000000": {}}, "V99999A", name="qc_config")


def test_require_config_keys():
    config = {"energy_param": "cuspEmax", "threshold": 900}
    require_config_keys(config, ["energy_param", "threshold"], "test config")

    with pytest.raises(
        ValueError,
        match=r"test config is missing required key\(s\) \['final_cut_field'\]",
    ):
        require_config_keys(config, ["energy_param", "final_cut_field"], "test config")


def test_expand_filelist_plain_files():
    assert expand_filelist(["b.lh5", "a.lh5", "b.lh5"]) == ["a.lh5", "b.lh5"]


def test_expand_filelist_from_filelist(tmp_path):
    filelist = tmp_path / "cal.filelist"
    filelist.write_text("b.lh5\n\na.lh5\n  \nb.lh5\n")
    assert expand_filelist([str(filelist)]) == ["a.lh5", "b.lh5"]


def test_expand_filelist_empty_args():
    with pytest.raises(ValueError, match="--tcm-files: no input files"):
        expand_filelist([], "--tcm-files")
    with pytest.raises(ValueError, match="no input files"):
        expand_filelist(None)


def test_expand_filelist_empty_filelist(tmp_path):
    filelist = tmp_path / "cal.filelist"
    filelist.write_text("\n  \n")
    with pytest.raises(ValueError, match="is empty"):
        expand_filelist([str(filelist)])


def test_check_pulser_mask():
    mask = np.array([True, False, True])
    check_pulser_mask(mask, np.array([True, True, False]), "ch1000000")

    with pytest.raises(ValueError, match="pulser mask length 3 != number of loaded"):
        check_pulser_mask(mask, np.array([True, False]), "ch1000000")


def test_get_pulser_mask_missing_mask_key(tmp_path):
    pulser_file = tmp_path / "pulser.yaml"
    pulser_file.write_text(yaml.dump({"idxs": [1, 2]}))
    with pytest.raises(KeyError, match="does not contain a 'mask' key"):
        get_pulser_mask(str(pulser_file))


@pytest.fixture
def configs_tree(tmp_path):
    (tmp_path / "config.yaml").write_text(
        yaml.dump({"snakemake_rules": {"tier_tcm": {"inputs": {"config": "c.yaml"}}}})
    )
    (tmp_path / "validity.yaml").write_text(
        yaml.dump([{"valid_from": "20230101T000000Z", "apply": ["config.yaml"]}])
    )
    return tmp_path


def test_get_rule_config(configs_tree):
    config = get_rule_config(configs_tree, "tier_tcm", "20230201T000000Z", "cal")
    assert config["inputs"]["config"] == "c.yaml"

    with pytest.raises(KeyError, match=r"no snakemake_rules\.tier_dsp entry"):
        get_rule_config(configs_tree, "tier_dsp", "20230201T000000Z", "cal")


def test_alias_table_membership(tmp_path):
    lh5_file = tmp_path / "test.lh5"
    with h5py.File(lh5_file, "w") as f:
        grp = f.create_group("det/raw_blind")
        grp.attrs["datatype"] = "table{a}"
        grp.create_dataset("a", data=[1, 2, 3])
        f["det"].attrs["datatype"] = "struct{raw_blind}"

    # 'raw' is a substring of 'raw_blind' and must still be registered
    alias_table(lh5_file, {"det/raw_blind": "det/raw"})

    with h5py.File(lh5_file) as f:
        fields = f["det"].attrs["datatype"]
        assert fields == "struct{raw_blind,raw}"
        assert list(f["det/raw/a"]) == [1, 2, 3]

    # re-registering an existing member must not duplicate it
    alias_table(lh5_file, {"det/raw_blind": "det/raw2"})
    with h5py.File(lh5_file) as f:
        assert f["det"].attrs["datatype"] == "struct{raw_blind,raw,raw2}"


def test_build_log_excepthook(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)

    log_file = tmp_path / "job.log"
    cfg = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"f": {"format": "%(levelname)s %(message)s"}},
        "handlers": {"dataflow": {"class": "logging.FileHandler", "formatter": "f"}},
    }
    log = build_log(cfg, str(log_file))

    sys.excepthook(ValueError, ValueError("boom"), None)
    for handler in logging.getLogger().handlers:
        handler.flush()

    text = log_file.read_text()
    assert "ERROR" in text
    assert "uncaught exception" in text
    assert "ValueError: boom" in text

    # KeyboardInterrupt is delegated to the default excepthook and must not
    # be logged as a second "uncaught exception" record (it may still reach
    # the log via the stderr redirection)
    sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert log_file.read_text().count("uncaught exception") == 1

    assert isinstance(log, logging.Logger)
