from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def check_input_files(files, argname="input") -> None:
    """Check that all input files exist, raising a named error otherwise.

    Parameters
    ----------
    files : str, pathlib.Path, list, or None
        Input file path(s) to check. ``None`` is a no-op (for optional
        arguments that were not provided).
    argname : str
        Name of the CLI argument, used in the error message.

    Raises
    ------
    FileNotFoundError
        Naming *argname* and every missing path (up to five, plus a count
        of any further ones).
    """
    if files is None:
        return
    if isinstance(files, str | Path):
        files = [files]
    missing = [str(file) for file in files if not Path(file).is_file()]
    if missing:
        shown = ", ".join(missing[:5])
        extra = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
        msg = f"{argname}: input file(s) not found: {shown}{extra}"
        raise FileNotFoundError(msg)


def expand_filelist(files, argname="--files", check_exists=True):
    """Expand a CLI file argument that may be a single ``.filelist`` file.

    Parameters
    ----------
    files : list of str or None
        Value of an ``nargs="*"`` argparse argument: either the input files
        themselves or a single ``.filelist`` file containing one path per
        line.
    argname : str
        Name of the CLI argument, used in error messages.
    check_exists : bool
        When ``True`` (the default) every resulting file path is checked
        for existence via :func:`check_input_files`.

    Returns
    -------
    list of str
        The sorted, de-duplicated input file paths.
    """
    if not files:
        msg = f"{argname}: no input files provided"
        raise ValueError(msg)
    if len(files) == 1 and Path(files[0]).suffix == ".filelist":
        with Path(files[0]).open() as f:
            expanded = [line.strip() for line in f if line.strip()]
        if not expanded:
            msg = f"{argname}: filelist {files[0]} is empty"
            raise ValueError(msg)
        files = expanded
    files = sorted(np.unique(files))
    if check_exists:
        check_input_files(files, argname)
    return files


def prepare_output_paths(*paths) -> None:
    """Create the parent directories of the given output paths.

    Call early in a script, before any expensive computation, so that an
    unwritable output location fails up front instead of after the work is
    done. ``None`` entries (optional outputs that were not requested) are
    skipped.
    """
    for path in paths:
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)


def parse_json_arg(value, argname):
    """Parse a JSON-encoded CLI argument, naming the argument on failure.

    Parameters
    ----------
    value : str
        The JSON string to parse.
    argname : str
        Name of the CLI argument, used in the error message.
    """
    try:
        return json.loads(value)
    except json.JSONDecodeError as err:
        msg = f"{argname} is not valid JSON: {err}"
        raise ValueError(msg) from None
