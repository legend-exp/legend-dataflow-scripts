"""Tests for legenddataflowscripts.utils.plot_dict.plot_dict_to_lgdo."""

from __future__ import annotations

import lh5
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from legenddataflowscripts.utils import (
    data_for,
    fill_plot_dict,
    lh5_key,
    plot_dict_to_lgdo,
)


def test_lh5_key():
    assert lh5_key(2614.511) == "2614p511"
    assert lh5_key("0.5") == "0p5"
    assert lh5_key("2614_stability") == "2614_stability"


def test_plot_dict_to_lgdo_round_trip(tmp_path):
    fig = plt.figure()
    plot_dict = {
        "ecal": {
            "cuspEmax_ctc_cal": {
                "peak_fits": fig,
                "peak_hists": {
                    2614.511: {"counts": np.arange(4), "edges": np.linspace(0, 1, 5)}
                },
                "2614_timemap": {"counts": np.ones((3, 2)), "time_edges": [0, 1, 2, 3]},
            },
            "empty": {},
        },
        "nopt": {"best_par": np.float64(4.5), "func": "gauss_on_step", "ok": True},
        "skipped": None,
    }
    plt.close(fig)

    struct = plot_dict_to_lgdo(plot_dict)
    assert set(struct) == {"ecal", "nopt"}
    assert set(struct["ecal"]) == {"cuspEmax_ctc_cal"}
    assert "peak_fits" not in struct["ecal"]["cuspEmax_ctc_cal"]

    path = str(tmp_path / "plt.lh5")
    lh5.write(struct, "V00000A", path, wo_mode="append")
    out = lh5.read("V00000A", path)
    param = out["ecal"]["cuspEmax_ctc_cal"]
    np.testing.assert_array_equal(
        param["peak_hists"]["2614p511"]["counts"].nda, np.arange(4)
    )
    assert param["2614_timemap"]["counts"].nda.shape == (3, 2)
    assert out["nopt"]["best_par"].value == 4.5
    assert out["nopt"]["ok"].value


def test_plot_dict_to_lgdo_nothing():
    assert plot_dict_to_lgdo({"fig": plt.figure()}) is None
    plt.close("all")


def _plot(obj, data, scale=1, figsize=(4, 3)):  # noqa: ARG001
    return "figure"


def _data(obj, data, scale=1):  # noqa: ARG001
    return {"values": np.asarray(data) * scale}


def test_data_for_pairing():
    assert data_for(_plot, None, [1, 2]) is None  # no data_func yet
    _plot.data_func = _data
    try:
        out = data_for(_plot, None, [1, 2], scale=3, figsize=(8, 6))  # figsize dropped
        np.testing.assert_array_equal(out["values"], [3, 6])
    finally:
        del _plot.data_func


def test_fill_plot_dict_stores_data():
    _plot.data_func = _data
    try:
        opts = {"spec": {"function": _plot, "options": {"scale": 2}},
                "other": {"function": lambda *_: "fig", "options": None}}  # fmt: skip
        out = fill_plot_dict(None, [1], opts)
    finally:
        del _plot.data_func
    assert set(out) == {"spec", "spec_data", "other"}
    np.testing.assert_array_equal(out["spec_data"]["values"], [2])
