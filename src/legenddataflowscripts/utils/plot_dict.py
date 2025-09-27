from __future__ import annotations


def fill_plot_dict(plot_class, data, plot_options, plot_dict=None):
    if plot_dict is None:
        plot_dict = {}
    if plot_options is not None:
        for key, item in plot_options.items():
            if item["options"] is not None:
                plot_dict[key] = item["function"](plot_class, data, **item["options"])
            else:
                plot_dict[key] = item["function"](plot_class, data)
    return plot_dict
