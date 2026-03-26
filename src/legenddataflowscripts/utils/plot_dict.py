from __future__ import annotations


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
        Updated *plot_dict* with one entry per key in *plot_options*.
    """
    if plot_dict is None:
        plot_dict = {}
    if plot_options is not None:
        for key, item in plot_options.items():
            if item["options"] is not None:
                plot_dict[key] = item["function"](plot_class, data, **item["options"])
            else:
                plot_dict[key] = item["function"](plot_class, data)
    return plot_dict
