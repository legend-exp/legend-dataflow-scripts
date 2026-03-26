"""Utility helpers for the LEGEND dataflow Snakemake workflow.

This module provides functions for variable substitution in configuration
objects, dynamic Snakemake rule renaming, and read-only filesystem path
translation.  Most of these utilities are re-exported from the top-level
:mod:`legenddataflowscripts` package for convenience.
"""

from __future__ import annotations

import copy
import os
import re
import string
from pathlib import Path


def subst_vars_impl(x, var_values, ignore_missing=False):
    """Recursively substitute ``$VAR`` placeholders in a nested structure.

    Traverses *x* depth-first.  Any string value containing ``$`` is treated
    as a :class:`string.Template` and expanded using *var_values*.

    Parameters
    ----------
    x : str, dict, list, or other
        The value to process.  Strings are expanded in-place; dicts and lists
        are traversed recursively.  All other types are returned unchanged.
    var_values : dict
        Mapping of variable names to substitution values.
    ignore_missing : bool
        When ``True`` unknown placeholders are left as-is (uses
        :meth:`string.Template.safe_substitute`).  When ``False`` (default) an
        unknown placeholder raises :class:`KeyError`.

    Returns
    -------
    str, dict, list, or other
        *x* with all ``$VAR`` placeholders replaced.
    """
    if isinstance(x, str):
        if "$" in x:
            if ignore_missing:
                return string.Template(x).safe_substitute(var_values)
            return string.Template(x).substitute(var_values)
        return x
    if isinstance(x, dict):
        for key in x:
            value = x[key]
            new_value = subst_vars_impl(value, var_values, ignore_missing)
            if new_value is not value:
                x[key] = new_value
        return x
    if isinstance(x, list):
        for i in range(len(x)):
            value = x[i]
            new_value = subst_vars_impl(value, var_values, ignore_missing)
            if new_value is not value:
                x[i] = new_value
        return x
    return x


def subst_vars(
    props,
    var_values=None,
    use_env=False,
    ignore_missing=False,
):
    """Substitute ``$VAR`` placeholders in a configuration object.

    Thin wrapper around :func:`subst_vars_impl` that optionally merges the
    current process environment into the substitution table before expansion.
    Environment variables take lower priority than explicit entries in
    *var_values*.

    Parameters
    ----------
    props : str, dict, list, or other
        Configuration object to expand in-place.
    var_values : dict, optional
        Explicit variable-name → value mapping.  Takes precedence over
        environment variables.
    use_env : bool
        When ``True`` the current environment (:data:`os.environ`) is merged
        into the substitution table.  Defaults to ``False``.
    ignore_missing : bool
        Passed through to :func:`subst_vars_impl`.  Defaults to ``False``.

    Returns
    -------
    str, dict, list, or other
        *props* with all recognisable ``$VAR`` placeholders expanded.
    """
    if var_values is None:
        var_values = {}
    combined_var_values = var_values
    if use_env:
        combined_var_values = dict(iter(os.environ.items()))
        combined_var_values.update(copy.copy(var_values))

    return subst_vars_impl(props, combined_var_values, ignore_missing)


def subst_vars_in_snakemake_config(workflow, config):
    """Expand ``$VAR`` placeholders in a Snakemake workflow configuration dict.

    Reads the path of the first Snakemake config file, sets ``$_`` to its
    parent directory, and calls :func:`subst_vars` on *config* with environment
    variable expansion enabled.  Afterwards the ``execenv`` key is resolved to
    the entry matching ``config["system"]`` (falling back to ``"bare"``).

    This function is typically called at the top of a ``Snakefile``:

    .. code-block:: python

        from legenddataflowscripts.workflow import subst_vars_in_snakemake_config

        subst_vars_in_snakemake_config(workflow, config)

    Parameters
    ----------
    workflow : snakemake.workflow.Workflow
        Active Snakemake workflow object (provides access to config file paths).
    config : dict
        Snakemake configuration dictionary to expand in-place.

    Raises
    ------
    RuntimeError
        If no config file has been passed to Snakemake
        (``workflow.overwrite_configfiles`` is empty).
    """
    if len(workflow.overwrite_configfiles) == 0:
        msg = "configfile not set!"
        raise RuntimeError(msg)

    config_filename = workflow.overwrite_configfiles[0]

    subst_vars(
        config,
        var_values={"_": Path(config_filename).parent},
        use_env=True,
        ignore_missing=False,
    )
    if "execenv" in config:
        if "system" in config:
            config["execenv"] = config["execenv"][config["system"]]
        else:
            config["execenv"] = config["execenv"]["bare"]


def set_last_rule_name(workflow, new_name):
    """Sets the name of the most recently created rule to be `new_name`.
    Useful when creating rules dynamically (i.e. unnamed).

    Warning
    -------
    This could mess up the workflow. Use at your own risk.
    """
    rules = workflow._rules
    last_key = next(reversed(rules))
    assert last_key == rules[last_key].name

    rules[new_name] = rules.pop(last_key)
    rules[new_name].name = new_name

    if workflow.default_target == last_key:
        workflow.default_target = new_name

    if last_key in workflow._localrules:
        workflow._localrules.remove(last_key)
        workflow._localrules.add(new_name)

    workflow.check_localrules()


def as_ro(config, path):
    """Translate a path (or list of paths) to its read-only filesystem equivalent.

    Some HPC sites expose the same data under both a read-write and a
    read-only mount point.  When ``config["read_only_fs_sub_pattern"]`` is
    set to a two-element list ``[pattern, replacement]`` this function applies
    :func:`re.sub` to convert *path* to the read-only mount.  If the key is
    absent or ``None`` the original *path* is returned unchanged.

    Parameters
    ----------
    config : dict
        Workflow configuration dict.  Inspected for the optional key
        ``"read_only_fs_sub_pattern"``.
    path : str, pathlib.Path, or list
        The path or collection of paths to translate.

    Returns
    -------
    str, pathlib.Path, or list
        Translated path(s).  The return type mirrors the input type.
    """
    if (
        "read_only_fs_sub_pattern" not in config
        or config["read_only_fs_sub_pattern"] is None
    ):
        return path

    sub_pattern = config["read_only_fs_sub_pattern"]

    if isinstance(path, str):
        return re.sub(*sub_pattern, path)
    if isinstance(path, Path):
        return Path(re.sub(*sub_pattern, path.name))

    return [as_ro(config, p) for p in path]
