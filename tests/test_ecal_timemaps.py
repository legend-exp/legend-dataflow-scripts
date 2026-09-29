"""Tests for the ecal timemap binning functions."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from legenddataflowscripts.par.geds.hit import ecal

RNG = np.random.default_rng(1)
N = 2000


def _data():
    return pd.DataFrame(
        {
            "timestamp": 1.7e9 + RNG.uniform(0, 3600, N),
            "e_cal": np.r_[
                RNG.normal(2614.5, 1.2, N // 2), RNG.normal(1000, 0.3, N // 2)
            ],
            "bl_mean": RNG.normal(15000, 3, N),
            "is_pulser": np.r_[np.zeros(N // 2, bool), np.ones(N // 2, bool)],
            "is_valid": np.ones(N, bool),
        }
    )


def test_bin_timemaps():
    df = _data()
    for hist in (
        ecal.bin_2614_timemap(df, "e_cal", "is_valid"),
        ecal.bin_pulser_timemap(df, "e_cal", "is_valid"),
        ecal.bin_baseline_timemap(df),
    ):
        assert hist["counts"].shape == (
            len(hist["time_edges"]) - 1,
            len(hist["value_edges"]) - 1,
        )
    hist = ecal.bin_2614_timemap(df, "e_cal", "is_valid")
    in_range = df.e_cal.between(hist["value_edges"][0], hist["value_edges"][-1])
    assert hist["counts"].sum() == in_range.sum()
    assert ecal.bin_2614_timemap(df, "e_cal", "~is_valid") == {}


def test_plot_timemaps():
    df = _data()
    assert isinstance(ecal.plot_2614_timemap(df, "e_cal", "is_valid"), Figure)
    assert isinstance(ecal.plot_pulser_timemap(df, "e_cal", "is_valid"), Figure)
    assert isinstance(ecal.plot_baseline_timemap(df), Figure)
    assert isinstance(ecal.plot_2614_timemap(df, "e_cal", "~is_valid"), Figure)


def test_timemaps_paired():
    assert ecal.plot_2614_timemap.data_func is ecal.bin_2614_timemap
    assert ecal.plot_pulser_timemap.data_func is ecal.bin_pulser_timemap
    assert ecal.plot_baseline_timemap.data_func is ecal.bin_baseline_timemap
