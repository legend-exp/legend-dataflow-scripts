from __future__ import annotations

import logging
import os
import subprocess
import sys

import pytest
import yaml
from dbetto import AttrsDict

from legenddataflowscripts.workflow import execenv

os.environ["XDG_RUNTIME_DIR"] = "whatever"


@pytest.fixture(scope="module")
def config():
    return AttrsDict(
        {
            "paths": {"install": ".snakemake/software"},
            "execenv": {
                "cmd": "apptainer exec",
                "arg": "image.sif",
                "env": {
                    "VAR1": "val1",
                    "VAR2": "val2",
                },
            },
        }
    )


def test_execenv2str():
    assert (
        execenv._execenv2str(["cmd", "-v", "opt"], {"VAR1": "val1", "VAR2": "val2"})
        == "VAR1=val1 VAR2=val2 cmd -v opt"
    )


def test_execenv_prefix(config):
    cmd_expr, cmd_env = execenv.execenv_prefix(config, as_string=False)

    assert cmd_expr == [
        "apptainer",
        "exec",
        "--env=VAR1=val1",
        "--env=VAR2=val2",
        "--bind=whatever",
        "image.sif",
    ]
    assert cmd_env == config.execenv.env

    config.execenv.cmd = "docker run"
    cmd_expr, cmd_env = execenv.execenv_prefix(config, as_string=False)

    assert cmd_expr == [
        "docker",
        "run",
        "--env=VAR1=val1",
        "--env=VAR2=val2",
        "--volume=whatever:whatever",
        "image.sif",
    ]
    assert cmd_env == config.execenv.env

    config.execenv.cmd = "shifter"
    config.execenv.arg = "--image=legendexp/legend-base:latest"
    cmd_expr, cmd_env = execenv.execenv_prefix(config, as_string=False)

    assert cmd_expr == [
        "shifter",
        "--env=VAR1=val1",
        "--env=VAR2=val2",
        "--image=legendexp/legend-base:latest",
    ]
    assert cmd_env == config.execenv.env

    cmd_str = execenv.execenv_prefix(config, as_string=True)
    assert cmd_str == (
        "VAR1=val1 VAR2=val2 "
        "shifter --env=VAR1=val1 --env=VAR2=val2 "
        "--image=legendexp/legend-base:latest "
    )

    config = {
        "execenv": {
            "env": {
                "VAR1": "val1",
                "VAR2": "val2",
            }
        }
    }
    cmd_str = execenv.execenv_prefix(config, as_string=True)
    assert cmd_str == "VAR1=val1 VAR2=val2  "


def test_execenv_pyexe(config):
    cmd_str = execenv.execenv_pyexe(config, "dio-boe")

    assert cmd_str == (
        "VAR1=val1 VAR2=val2 "
        "shifter --env=VAR1=val1 --env=VAR2=val2 "
        "--image=legendexp/legend-base:latest "
        ".snakemake/software/bin/dio-boe "
    )


def test_execenv_prefix_pixi_mode():
    # a config without an execenv section is a pixi-managed cycle: no
    # container prefix, no env exports
    for cfg in ({}, {"paths": {"install": "x"}}, {"execenv": {}}, {"execenv": None}):
        assert execenv.execenv_prefix(cfg, as_string=True) == ""
        assert execenv.execenv_prefix(cfg, as_string=False) == ([], {})


def test_execenv_pyexe_pixi_mode():
    # ... and executables resolve from the ambient environment's PATH
    for cfg in ({}, {"paths": {"install": "x"}}, {"execenv": {}}):
        assert execenv.execenv_pyexe(cfg, "dio-boe") == "dio-boe "
        assert execenv.execenv_pyexe(cfg, "dio-boe", as_string=False) == (
            ["dio-boe"],
            {},
        )


def test_install_refuses_pixi_managed_cycle(tmp_path):
    cfg = tmp_path / "dataflow-config.yaml"
    cfg.write_text(yaml.dump({"paths": {"install": str(tmp_path / "venv")}}))

    args = type(
        "Args",
        (),
        {"config_file": str(cfg), "system": "bare", "remove": False, "editable": False},
    )()
    with pytest.raises(SystemExit, match="managed by pixi"):
        execenv.install(args)


def test_cmdexec_pixi_mode_runs_bare_command(monkeypatch, tmp_path):

    cfg = tmp_path / "dataflow-config.yaml"
    cfg.write_text(yaml.dump({"paths": {"install": str(tmp_path / "venv")}}))

    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        calls["kwargs"] = kwargs

    monkeypatch.setattr(subprocess, "run", fake_run)

    args = type(
        "Args",
        (),
        {"config_file": str(cfg), "system": "bare", "command": ["echo", "hi"]},
    )()
    execenv.cmdexec(args)

    assert calls["cmd"] == ["echo", "hi"]
    # no env override: the ambient (pixi) environment is used as-is
    assert "env" not in calls["kwargs"]


def test_install_warns_deprecation_with_execenv(monkeypatch, tmp_path, caplog):

    cfg = tmp_path / "dataflow-config.yaml"
    cfg.write_text(
        yaml.dump(
            {
                "paths": {"install": str(tmp_path / "venv")},
                "execenv": {"bare": {"env": {"VAR1": "val1"}}},
            }
        )
    )

    # stop before any real venv work happens
    def _stop(*_args, **_kwargs):
        msg = "stop-here"
        raise RuntimeError(msg)

    monkeypatch.setattr(execenv.subprocess, "run", _stop)

    args = type(
        "Args",
        (),
        {"config_file": str(cfg), "system": "bare", "remove": False, "editable": False},
    )()
    with (
        caplog.at_level(logging.WARNING, logger="legenddataflowscripts"),
        pytest.raises(RuntimeError, match="stop-here"),
    ):
        execenv.install(args)

    assert "deprecated" in caplog.text
    assert "pixi" in caplog.text


def test_dataflow_no_subcommand(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["dataflow", "-v"])
    with pytest.raises(SystemExit) as excinfo:
        execenv.dataflow()
    assert excinfo.value.code == 1
    assert "usage" in capsys.readouterr().err.lower()


def test_select_execenv_missing_system():
    config = AttrsDict({"execenv": {"bare": {"cmd": "x"}, "lngs": {"cmd": "y"}}})
    assert execenv._select_execenv(config, "bare", "cfg.yaml") == {"cmd": "x"}

    with pytest.raises(KeyError, match=r"not found under 'execenv' in cfg\.yaml"):
        execenv._select_execenv(config, "nersc", "cfg.yaml")
