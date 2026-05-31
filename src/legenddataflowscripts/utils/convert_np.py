from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def convert_dict_np_to_float(dic: dict) -> dict:
    """
    Convert numpy scalars in a dictionary to native Python types.

    Recursively converts all numpy scalar values (integers, floats, booleans)
    to their Python equivalents to ensure JSON/YAML serializability.

    Parameters
    ----------
    dic : dict
        The dictionary to convert.

    Returns
    -------
    dict
        The dictionary with all numpy scalars converted to Python types.
    """
    for key, value in dic.items():
        if isinstance(value, Mapping):
            dic[key] = convert_dict_np_to_float(value)
        elif isinstance(value, np.generic):
            dic[key] = value.item()
        elif isinstance(value, Sequence) and not isinstance(value, str):
            dic[key] = [x.item() if isinstance(x, np.generic) else x for x in value]
    return dic
