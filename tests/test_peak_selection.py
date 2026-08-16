"""Tests for the peak-selection helpers used by the dsp par scripts.

Covers ``build_peak_dicts`` (which drops peaks with no candidate events, so a
channel whose DAQ threshold sits above a configured gamma line does not hand
``None`` to ``build_dsp``) and ``require_peaks_present`` (which makes the peak
file consumers fail loudly instead of filtering down to zero rows).
"""

from __future__ import annotations

import numpy as np
import pytest

from legenddataflowscripts.par.geds.dsp.evtsel import build_peak_dicts
from legenddataflowscripts.utils import require_peaks_present

PEAKS = [583.191, 727.33, 860.564, 1620.5, 2614.553]
KEV_WIDTHS = [(25, 30), (25, 35), (25, 40), (15, 55), (70, 70)]


def _masks(counts):
    return {
        peak: np.arange(count, dtype=int)
        for peak, count in zip(PEAKS, counts, strict=True)
    }


def test_all_peaks_populated():
    counts = [33123, 10172, 6538, 2519, 12999]
    pk_dicts = build_peak_dicts(PEAKS, KEV_WIDTHS, _masks(counts))

    assert list(pk_dicts) == PEAKS
    for peak, kev_width, count in zip(PEAKS, KEV_WIDTHS, counts, strict=True):
        entry = pk_dicts[peak]
        assert entry["kev_width"] == kev_width
        assert len(entry["idxs"][0]) == count
        assert entry["n_rows_read"] == 0
        assert entry["obj_buf_start"] == 0
        assert entry["obj_buf"] is None


def test_empty_peaks_are_skipped():
    # V01404A in p18: the DAQ threshold sits above the three lowest lines
    counts = [0, 0, 0, 7315, 12112]
    pk_dicts = build_peak_dicts(PEAKS, KEV_WIDTHS, _masks(counts))

    assert list(pk_dicts) == [1620.5, 2614.553]
    assert len(pk_dicts[1620.5]["idxs"][0]) == 7315
    assert pk_dicts[1620.5]["kev_width"] == (15, 55)


def test_no_peaks_at_all():
    assert build_peak_dicts(PEAKS, KEV_WIDTHS, _masks([0] * 5)) == {}


def test_skipped_peaks_are_logged():
    class _Recorder:
        def __init__(self):
            self.messages = []

        def warning(self, msg):
            self.messages.append(msg)

    log = _Recorder()
    build_peak_dicts(PEAKS, KEV_WIDTHS, _masks([0, 0, 0, 7315, 12112]), log=log)

    assert len(log.messages) == 3
    assert all("skipping this peak" in msg for msg in log.messages)
    assert "583.191" in log.messages[0]


def test_require_peaks_present_passes():
    require_peaks_present(np.array([1620, 2614]), [2614], "peak file")


def test_require_peaks_present_lists_all_missing():
    with pytest.raises(ValueError, match=r"missing required peak\(s\) \[583, 727\]"):
        require_peaks_present(np.array([1620, 2614]), [583, 727, 2614], "peak file")


def test_require_peaks_present_names_the_context():
    with pytest.raises(ValueError, match=r"my-peaks\.lh5"):
        require_peaks_present(np.array([2614]), [583], "peak file my-peaks.lh5")
