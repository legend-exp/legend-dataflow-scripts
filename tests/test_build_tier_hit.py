"""Tests for the build-tier-hit float32 parameter cast.

Calibration parameters load from the pars YAML as python (64-bit) floats,
which makes ``Table.eval`` promote every derived hit column to float64 even
when the DSP inputs are float32.  ``_cast_op_params_f32`` casts them at load
time; these tests cover the helper and the end-to-end effect through the
``build-tier-hit`` entry point on a synthetic float32 DSP file.
"""

from __future__ import annotations

import json
import sys
import warnings

import lh5
import numpy as np
import pytest
import yaml
from lgdo import Array, Table

from legenddataflowscripts.tier.hit import _cast_op_params_f32, build_tier_hit

CHANNEL = "ch1084803"


def test_cast_op_params_f32_types():
    cfg = {
        "operations": {
            "x_cal": {
                "expression": "a + b*x",
                "parameters": {
                    "a": 0.5,
                    "b": 2,
                    "flag": True,
                    "name": "cal",
                    "poly": [1.0, 2.5, 3],
                },
            },
            "no_pars": {"expression": "x*x"},
        }
    }
    out = _cast_op_params_f32(cfg)
    pars = out["operations"]["x_cal"]["parameters"]
    assert pars["a"].dtype == np.float32
    # ints, bools and strings are left untouched
    assert pars["b"] == 2
    assert isinstance(pars["b"], int)
    assert pars["flag"] is True
    assert pars["name"] == "cal"
    # floats inside lists are cast, ints preserved
    assert pars["poly"][0].dtype == np.float32
    assert isinstance(pars["poly"][2], int)


@pytest.fixture
def workspace(tmp_path):
    """Synthetic f32 DSP file + minimal rule-config tree for tier_hit."""
    dsp = tmp_path / "dsp.lh5"
    rng = np.random.default_rng(42)
    energies = rng.uniform(100, 3000, 500).astype("float32")
    lh5.LH5Store().write(
        Table(col_dict={"cuspEmax": Array(energies)}),
        "dsp",
        str(dsp),
        group=CHANNEL,
        wo_mode="overwrite_file",
    )

    cfgdir = tmp_path / "configs"
    cfgdir.mkdir()
    (cfgdir / "hit_config.yaml").write_text(
        yaml.safe_dump(
            {
                "outputs": ["cuspEmax_cal"],
                "operations": {
                    "cuspEmax_cal": {
                        "expression": "a + b*cuspEmax",
                        "parameters": {"a": 0.25, "b": 0.13},
                    }
                },
            }
        )
    )
    (cfgdir / "logging.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "disable_existing_loggers": False,
                "handlers": {
                    "dataflow": {
                        "class": "logging.FileHandler",
                        "level": "INFO",
                        "filename": "__auto__",
                        "mode": "a",
                    }
                },
                "loggers": {
                    "prod": {"level": "INFO", "handlers": ["dataflow"]},
                },
            }
        )
    )
    (cfgdir / "cfg.yaml").write_text(
        yaml.safe_dump(
            {
                "snakemake_rules": {
                    "tier_hit": {
                        "inputs": {"hit_config": {"__default__": "$_/hit_config.yaml"}},
                        "options": {"logging": "$_/logging.yaml", "logger": "prod"},
                    }
                }
            }
        )
    )
    (cfgdir / "validity.yaml").write_text(
        yaml.safe_dump(
            [
                {
                    "valid_from": "20220101T000000Z",
                    "mode": "reset",
                    "apply": ["cfg.yaml"],
                },
                {
                    "valid_from": "20220101T000000Z",
                    "category": "cal",
                    "mode": "reset",
                    "apply": ["cfg.yaml"],
                },
            ]
        )
    )
    return {"dsp": dsp, "configs": cfgdir, "energies": energies, "tmp": tmp_path}


def test_build_tier_hit_outputs_float32(workspace, monkeypatch):
    out = workspace["tmp"] / "hit.lh5"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build-tier-hit",
            "--input",
            str(workspace["dsp"]),
            "--configs",
            str(workspace["configs"]),
            "--table-map",
            json.dumps({CHANNEL: f"{CHANNEL}/dsp"}),
            "--datatype",
            "cal",
            "--timestamp",
            "20230101T000000Z",
            "--tier",
            "hit",
            "--output",
            str(out),
            "--log",
            str(workspace["tmp"] / "hit.log"),
        ],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build_tier_hit()

    col = lh5.read(f"{CHANNEL}/hit/cuspEmax_cal", str(out))
    # float parameters were cast, so the f32 input column stays f32
    assert col.nda.dtype == np.dtype("float32")
    ref = 0.25 + 0.13 * workspace["energies"].astype("float64")
    np.testing.assert_allclose(col.nda, ref, rtol=5e-6)
