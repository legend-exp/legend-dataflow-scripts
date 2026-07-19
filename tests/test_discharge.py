"""Tests for legenddataflowscripts.utils.discharge.get_is_recovering_mask."""

from __future__ import annotations

import numpy as np
import pytest

from legenddataflowscripts.utils import get_is_recovering_mask


def _reference_loop(timestamps, discharge_timestamps, window=0.01):
    """Verbatim structure of the original per-discharge loop."""
    is_recovering = np.full(len(timestamps), False, dtype=bool)
    for tstamp in discharge_timestamps:
        is_recovering = is_recovering | np.where(
            (((timestamps - tstamp) < window) & ((timestamps - tstamp) > 0)),
            True,
            False,
        )
    return is_recovering


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_loop_on_random_data(seed):
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.uniform(1.7e9, 1.7e9 + 20, 5000))
    discharges = rng.choice(ts, 40, replace=False)
    ref = _reference_loop(ts, discharges)
    fast = get_is_recovering_mask(ts, discharges)
    assert np.array_equal(fast, ref)
    # discharges must actually flag something for the test to be meaningful
    assert np.count_nonzero(ref) > 0


def test_edge_cases():
    ts = np.array([10.0, 10.005, 10.01, 10.02])
    # no discharges
    assert np.array_equal(
        get_is_recovering_mask(ts, np.array([])), np.zeros(4, dtype=bool)
    )
    # event exactly at the discharge time is NOT recovering (t - d == 0)
    d = np.array([10.0])
    assert np.array_equal(get_is_recovering_mask(ts, d), _reference_loop(ts, d))
    # duplicate and unsorted discharges
    d = np.array([10.02, 10.0, 10.0])
    assert np.array_equal(get_is_recovering_mask(ts, d), _reference_loop(ts, d))


def test_production_call_pattern_stays_all_false():
    """The call sites currently pass ``np.where(ts[discharges])[0]`` (indices,
    not timestamps); with epoch-scale event timestamps the mask is all-False.
    The vectorized form must reproduce that exactly."""
    rng = np.random.default_rng(3)
    ts = np.sort(rng.uniform(1.7e9, 1.7e9 + 3600, 2000))
    discharges = rng.random(2000) < 0.01
    discharge_timestamps = np.where(ts[discharges])[0]
    ref = _reference_loop(ts, discharge_timestamps)
    fast = get_is_recovering_mask(ts, discharge_timestamps)
    assert np.array_equal(fast, ref)
    assert not fast.any()
