from __future__ import annotations

import numpy as np
from dbetto.catalog import Props


def get_pulser_mask(pulser_file):
    """Load and concatenate pulser event masks from one or more files.

    Each file is expected to be a JSON or YAML file with a top-level ``mask``
    key containing a boolean array.  When multiple files are provided the
    individual masks are concatenated in order.

    Parameters
    ----------
    pulser_file : str or list of str
        Path or list of paths to pulser mask files.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(N,)`` where ``True`` marks pulser events.
    """
    if not isinstance(pulser_file, list):
        pulser_file = [pulser_file]
    mask = np.array([], dtype=bool)
    for file in pulser_file:
        pulser_dict = Props.read_from(file)
        pulser_mask = np.array(pulser_dict["mask"])
        mask = np.append(mask, pulser_mask)

    return mask
