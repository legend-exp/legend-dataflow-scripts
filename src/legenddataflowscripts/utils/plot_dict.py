from __future__ import annotations

import inspect

import numpy as np
from lgdo import Array, Scalar, Struct


def data_for(plot_func, *args, **options):
    """Run the data function paired with a plot function, if there is one.

    Plot functions that have a data counterpart carry it as ``data_func``.
    It is called with the same arguments; options it does not accept (e.g.
    ``figsize``) are dropped.

    Parameters
    ----------
    plot_func : callable
        Plot function from a ``plot_options`` entry.
    *args
        Positional arguments passed to the plot function.
    **options
        Keyword options from the ``plot_options`` entry.

    Returns
    -------
    dict or None
        The plot data, or ``None`` if *plot_func* has no data function.
    """
    data_func = getattr(plot_func, "data_func", None)
    if data_func is None:
        return None
    params = inspect.signature(data_func).parameters
    if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
        options = {k: v for k, v in options.items() if k in params}
    return data_func(*args, **options)


def fill_plot_dict(plot_class, data, plot_options, plot_dict=None):
    """Populate a dictionary with figures produced by calibration plot functions.

    Iterates over *plot_options* and, for each entry, calls the specified
    function with *plot_class* and *data* as positional arguments followed by
    any keyword arguments defined in ``item["options"]``.  Results are stored
    in *plot_dict* under the corresponding key.

    Parameters
    ----------
    plot_class : object
        Calibration class instance passed as the first argument to each plot
        function (e.g. a ``CalAoE`` or ``LQCal`` instance).
    data : pandas.DataFrame
        Event-level data passed as the second argument to each plot function.
    plot_options : dict or None
        Mapping of ``{label: {"function": callable, "options": dict | None}}``.
        If ``None`` or empty no figures are generated.
    plot_dict : dict, optional
        Existing dictionary to append results to.  A new empty dict is created
        when not provided.

    Returns
    -------
    dict
        Updated *plot_dict* with one entry per key in *plot_options*, plus
        ``<key>_data`` for plot functions with a paired data function (see
        :func:`data_for`).
    """
    if plot_dict is None:
        plot_dict = {}
    if plot_options is not None:
        for key, item in plot_options.items():
            options = item["options"] or {}
            plot_dict[key] = item["function"](plot_class, data, **options)
            plot_data = data_for(item["function"], plot_class, data, **options)
            if plot_data is not None:
                plot_dict[f"{key}_data"] = plot_data
    return plot_dict


def lh5_key(key) -> str:
    """LH5-safe field name: lgdo splits on ``.`` so ``2614.511`` -> ``2614p511``."""
    return str(key).replace("/", "_").replace(".", "p")


def plot_dict_to_lgdo(plot_dict):
    """Convert a nested plot dictionary to an lgdo ``Struct``, dropping figures.

    Dicts become ``Struct``, numeric arrays and lists become ``Array`` (any
    dimensionality) and numbers, bools and strings become ``Scalar``. Anything
    else (matplotlib figures, ``None``, empty containers) is skipped, so a
    plot dict still holding figures converts to its data alone.

    Parameters
    ----------
    plot_dict : dict
        Plot dictionary as written to the per-channel plot pickles.

    Returns
    -------
    lgdo.Struct or None
        The converted structure, or ``None`` if nothing convertible was found.
    """

    def convert(obj):
        if isinstance(obj, dict):
            fields = {}
            for key, val in obj.items():
                out = convert(val)
                if out is not None:
                    fields[lh5_key(key)] = out
            return Struct(fields) if fields else None
        if isinstance(obj, str | bool | int | float | np.number | np.bool_):
            return Scalar(obj)
        if isinstance(obj, list | tuple):
            obj = np.asarray(obj)
        if isinstance(obj, np.ndarray) and obj.size and obj.dtype.kind in "biuf":
            return Array(obj)
        return None

    return convert(plot_dict)
