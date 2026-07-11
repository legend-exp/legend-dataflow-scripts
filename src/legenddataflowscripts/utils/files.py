from __future__ import annotations

from pathlib import Path

import numpy as np


def expand_filelist(files, argname="--files"):
    """Expand a CLI file argument that may be a single ``.filelist`` file.

    Parameters
    ----------
    files : list of str or None
        Value of an ``nargs="*"`` argparse argument: either the input files
        themselves or a single ``.filelist`` file containing one path per
        line.
    argname : str
        Name of the CLI argument, used in error messages.

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
    return sorted(np.unique(files))
