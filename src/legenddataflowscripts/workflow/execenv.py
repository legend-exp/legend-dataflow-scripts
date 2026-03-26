from __future__ import annotations

import argparse
import logging
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

import colorlog
import dbetto
from dbetto import AttrsDict

from . import utils

log = logging.getLogger(__name__)

OCI_ENGINES = ["docker", "podman", "podman-hpc", "shifter"]


def _execenv2str(cmd_expr: Iterable, cmd_env: Mapping) -> str:
    return " ".join([f"{k}={v}" for k, v in cmd_env.items()]) + " " + " ".join(cmd_expr)


def apptainer_env_vars(cmdenv: Mapping) -> list[str]:
    return [f"--env={var}={val}" for var, val in cmdenv.items()]


def oci_engine_env_vars(cmdenv: Mapping) -> list[str]:
    # same syntax
    return apptainer_env_vars(cmdenv)


def execenv_prefix(
    config: AttrsDict, as_string: bool = True
) -> str | tuple[list, dict]:
    """Return the software environment command prefix.

    Builds the command-line prefix (e.g. ``apptainer run image.sif``) and the
    associated environment variable mapping from the ``execenv`` section of
    *config*.  Supported container runtimes:

    * **Apptainer / Singularity** - environment variables are passed via
      ``--env=KEY=VAL`` flags and the XDG runtime directory is bind-mounted if
      present.
    * **OCI engines** (Docker, Podman, podman-hpc, Shifter) - environment
      variables are passed via ``--env=KEY=VAL`` flags; the XDG runtime
      directory is volume-mounted for all engines except Shifter.

    Parameters
    ----------
    config : dbetto.AttrsDict
        Workflow configuration containing an optional ``execenv`` key with
        sub-keys ``cmd`` (container command), ``arg`` (container image/args),
        and ``env`` (extra environment variables).
    as_string : bool
        When ``True`` (default) a single space-separated string with a
        trailing space is returned.  When ``False`` a ``(cmdline, cmdenv)``
        tuple is returned for programmatic use.

    Returns
    -------
    str or (list, dict)
        The command prefix as a string (with trailing space) or as a
        ``(cmdline_list, env_dict)`` tuple.

    Note
    ----
    If *as_string* is ``True``, a space is appended to the returned string.
    """
    config = AttrsDict(config)

    cmdline = []
    cmdenv = {}
    if "execenv" in config and "env" in config.execenv:
        cmdenv |= config.execenv.env

    if "execenv" in config and "cmd" in config.execenv and "arg" in config.execenv:
        cmdline = shlex.split(config.execenv.cmd)

        has_xdg = False
        xdg_runtime_dir = os.getenv("XDG_RUNTIME_DIR")
        if xdg_runtime_dir:
            has_xdg = True

        if "env" in config.execenv:
            if any(exe in config.execenv.cmd for exe in ("apptainer", "singularity")):
                cmdline += apptainer_env_vars(config.execenv.env)
                if has_xdg:
                    cmdline += [f"--bind={xdg_runtime_dir}"]

            elif any(engine in config.execenv.cmd for engine in OCI_ENGINES):
                cmdline += oci_engine_env_vars(config.execenv.env)

            # no XDG mount with shifter
            if (
                any(exe in config.execenv.cmd for exe in OCI_ENGINES)
                and has_xdg
                and "shifter" not in config.execenv.cmd
            ):
                cmdline += [f"--volume={xdg_runtime_dir}:{xdg_runtime_dir}"]

        # now we can add the arguments
        _arg = config.execenv.arg
        cmdline += shlex.split(_arg if isinstance(_arg, str) else " ".join(_arg))

    if as_string:
        return _execenv2str(cmdline, cmdenv) + " "

    return cmdline, cmdenv


def execenv_pyexe(
    config: AttrsDict, exename: str, as_string: bool = True
) -> str | tuple[list, dict]:
    """Return the full command to invoke a virtualenv executable inside the container.

    Extends the container prefix from :func:`execenv_prefix` with the
    absolute path ``{config.paths.install}/bin/{exename}``.  Example result:
    ``apptainer run image.sif /opt/sw/bin/par-geds-dsp-pz``

    Parameters
    ----------
    config : dbetto.AttrsDict
        Workflow configuration.  Must have a ``paths.install`` key pointing to
        the root of the Python virtual environment.
    exename : str
        Name of the executable inside the virtualenv ``bin`` directory (e.g.
        ``"par-geds-dsp-pz"``).
    as_string : bool
        When ``True`` (default) a single space-separated string with a
        trailing space is returned.  When ``False`` a ``(cmdline, cmdenv)``
        tuple is returned.

    Returns
    -------
    str or (list, dict)
        The full command as a string (with trailing space) or as a
        ``(cmdline_list, env_dict)`` tuple.

    Note
    ----
    If *as_string* is ``True``, a space is appended to the returned string.
    """
    config = AttrsDict(config)

    cmdline, cmdenv = execenv_prefix(config, as_string=False)
    cmdline.append(f"{config.paths.install}/bin/{exename}")

    if as_string:
        return _execenv2str(cmdline, cmdenv) + " "

    return cmdline, cmdenv


def dataflow() -> None:
    """dataflow's CLI for installing and loading the software in the data production environment.

    .. code-block:: console

      $ dataflow --help
      $ dataflow install --help  # help section for a specific sub-command
    """

    parser = argparse.ArgumentParser(
        prog="dataflow", description="dataflow's command-line interface"
    )

    parser.add_argument(
        "-v", "--verbose", help="increase verbosity", action="store_true"
    )

    subparsers = parser.add_subparsers()

    parser_install = subparsers.add_parser(
        "install", help="install user software in data production environment"
    )
    parser_install.add_argument(
        "config_file", help="production cycle configuration file"
    )
    parser_install.add_argument(
        "-s",
        "--system",
        help="system running on",
        default="bare",
        type=str,
        required=False,
    )
    parser_install.add_argument(
        "-r",
        "--remove",
        help="remove software directory before installing software",
        action="store_true",
    )
    parser_install.add_argument(
        "-e",
        "--editable",
        help="install software with pip's --editable flag",
        action="store_true",
    )
    parser_install.set_defaults(func=install)

    parser_exec = subparsers.add_parser(
        "exec", help="load data production environment and execute a given command"
    )
    parser_exec.add_argument(
        "config_file", help="production cycle configuration file", type=str
    )
    parser_exec.add_argument(
        "-s",
        "--system",
        help="system running on",
        default="bare",
        type=str,
        required=False,
    )
    parser_exec.add_argument(
        "command", help="command to run within the container", type=str, nargs="+"
    )
    parser_exec.set_defaults(func=cmdexec)

    if len(sys.argv) < 2:
        parser.print_usage(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.verbose:
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s%(name)s [%(levelname)s] %(message)s"
            )
        )

        logger = logging.getLogger("legenddataflowscripts")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

    if args.func:
        args.func(args)


def install(args) -> None:
    """Install user software in the data production environment.

    Creates a Python virtual environment at ``config.paths.install`` (inside
    the container if one is configured), upgrades ``pip``, installs ``uv``,
    and then uses ``uv pip install`` to install the workflow root directory
    (the directory containing the config file) as the package source.

    .. code-block:: console

       $ dataflow install config.yaml
       $ dataflow install --editable config.yaml   # editable install
       $ dataflow install --remove  config.yaml    # wipe venv before installing

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (provided by :func:`dataflow`).
    """
    config_dict = AttrsDict(dbetto.utils.load_dict(args.config_file))
    config_loc = Path(args.config_file).resolve().parent

    utils.subst_vars(
        config_dict, var_values={"_": config_loc}, use_env=True, ignore_missing=False
    )

    config_dict["execenv"] = config_dict.execenv[args.system]

    # path to virtualenv location
    path_install = config_dict.paths.install

    if args.remove and Path(path_install).exists():
        msg = f"removing: {path_install}"
        log.info(msg)
        shutil.rmtree(path_install)

    def _runcmd(cmd_expr, cmd_env, **kwargs):
        msg = "running: " + _execenv2str(cmd_expr, cmd_env)
        log.debug(msg)

        subprocess.run(cmd_expr, env=os.environ | cmd_env, check=True, **kwargs)

    cmd_prefix, cmd_env = execenv_prefix(config_dict, as_string=False)
    # HACK: get the full path to this python interpreter in case there is no execenv prefix
    python = sys.executable if cmd_prefix == [] else "python"
    python_venv, _ = execenv_pyexe(config_dict, "python", as_string=False)

    # we'll use uv from the virtualenv (installed below)
    uv_expr = [*python_venv, "-m", "uv"]  # , "--quiet"

    # otherwise use python-venv
    cmd_expr = [*cmd_prefix, python, "-m", "venv", path_install]

    msg = f"configuring virtual environment in {path_install}"
    log.info(msg)
    _runcmd(cmd_expr, cmd_env)

    cmd_expr = [
        *python_venv,
        "-m",
        "pip",
        "--quiet",
        "--no-cache-dir",
        "install",
        "--upgrade",
        "--",
        "pip",
    ]

    log.info("upgrading pip")
    _runcmd(cmd_expr, cmd_env)

    # install uv
    cmd_expr = [
        *python_venv,
        "-m",
        "pip",
        "--quiet",
        "--no-cache-dir",
        "install",
        "--no-warn-script-location",
        "--",
        "uv",
    ]

    log.info("installing uv")
    _runcmd(cmd_expr, cmd_env)

    # and finally install legenddataflow with all dependencies
    # this must be done within the execenv, since jobs will be run within it

    cmd_expr = [
        *uv_expr,
        "pip",
        "--no-cache",
        "install",
        "--prefix",
        path_install,
        str(config_loc),
    ]
    if args.editable:
        cmd_expr.insert(-1, "--editable")

    log.info("installing packages")
    _runcmd(cmd_expr, cmd_env)


def cmdexec(args) -> None:
    """Load the data production environment and execute a given command.

    Prepends the container prefix (if any) to *args.command*, adds the
    virtualenv ``bin`` directory to ``PATH``, and runs the resulting command
    with :func:`subprocess.run`.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (provided by :func:`dataflow`).
        Must include ``config_file``, ``system``, and ``command``.
    """
    config_dict = AttrsDict(dbetto.utils.load_dict(args.config_file))
    config_loc = Path(args.config_file).resolve().parent

    utils.subst_vars(
        config_dict,
        var_values={"_": config_loc},
        use_env=True,
        ignore_missing=False,
    )
    config_dict["execenv"] = config_dict["execenv"][args.system]

    exe_path = Path(config_dict.paths.install).resolve() / "bin"

    cmd_prefix, cmd_env = execenv_prefix(config_dict, as_string=False)
    cmd_expr = [*cmd_prefix, *args.command]

    msg = "running: " + _execenv2str(cmd_expr, cmd_env)
    log.debug(msg)

    env_dict = os.environ | cmd_env
    env_dict["PATH"] = (
        f"{exe_path}:{env_dict['PATH']}"  # prepend the virtualenv bin dir
    )
    subprocess.run(cmd_expr, env=env_dict, check=True)
