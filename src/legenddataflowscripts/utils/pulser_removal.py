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
    masks = [np.array([], dtype=bool)]
    for file in pulser_file:
        pulser_dict = Props.read_from(file)
        if "mask" not in pulser_dict:
            msg = f"pulser file {file} does not contain a 'mask' key"
            raise KeyError(msg)
        masks.append(np.array(pulser_dict["mask"]))

    return np.concatenate(masks)


def check_pulser_mask(mask, threshold_mask, context) -> None:
    """Validate that the pulser mask matches the loaded events.

    ``mask`` (from the pulser files) must have one entry per event read from
    the input files, i.e. the same length as the ``threshold_mask``
    selection returned by ``load_data`` (only ``len()`` is used, so any
    sized object is accepted).

    Parameters
    ----------
    mask
        Pulser mask array.
    threshold_mask
        Event selection mask (or any sized object with one entry per event).
    context
        Table/channel name used in the error message.
    """
    if len(mask) != len(threshold_mask):
        msg = (
            f"pulser mask length {len(mask)} != number of loaded events "
            f"{len(threshold_mask)} for {context}; the pulser files and "
            "input filelists likely cover different runs"
        )
        raise ValueError(msg)
