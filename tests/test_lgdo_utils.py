"""Tests for legenddataflowscripts.utils.lgdo_utils.take_table_rows."""

from __future__ import annotations

import numpy as np
import pytest
from lgdo import Array, ArrayOfEqualSizedArrays, Table, VectorOfVectors, WaveformTable

from legenddataflowscripts.utils import take_table_rows

RNG = np.random.default_rng(11)
N = 20


def _make_table():
    wf = WaveformTable(
        t0=np.zeros(N),
        t0_units="ns",
        dt=np.full(N, 16.0),
        dt_units="ns",
        values=RNG.normal(0, 1, (N, 32)).astype("float32"),
        values_units="ADC",
    )
    return Table(
        col_dict={
            "waveform": wf,
            "daqenergy": Array(RNG.integers(0, 5000, N).astype("uint32")),
            "matrix": ArrayOfEqualSizedArrays(nda=RNG.normal(0, 1, (N, 4))),
        }
    )


@pytest.mark.parametrize(
    "idx",
    [
        np.array([0, 3, 7, 19]),
        RNG.random(N) < 0.4,
    ],
    ids=["int-index", "bool-mask"],
)
def test_take_table_rows_matches_fancy_indexing(idx):
    tbl = _make_table()
    sub = take_table_rows(tbl, idx)

    n_sel = int(np.count_nonzero(idx)) if idx.dtype == bool else len(idx)
    assert len(sub) == n_sel
    assert isinstance(sub["waveform"], WaveformTable)
    assert np.array_equal(sub["waveform"].values.nda, tbl["waveform"].values.nda[idx])
    assert np.array_equal(sub["waveform"].dt.nda, tbl["waveform"].dt.nda[idx])
    assert np.array_equal(sub["waveform"].t0.nda, tbl["waveform"].t0.nda[idx])
    assert sub["waveform"].dt.attrs["units"] == "ns"
    assert sub["waveform"].values.attrs["units"] == "ADC"
    assert np.array_equal(sub["daqenergy"].nda, tbl["daqenergy"].nda[idx])
    assert sub["daqenergy"].nda.dtype == np.uint32
    assert isinstance(sub["matrix"], ArrayOfEqualSizedArrays)
    assert np.array_equal(sub["matrix"].nda, tbl["matrix"].nda[idx])


def test_take_table_rows_rejects_unsupported_columns():
    tbl = Table(
        col_dict={
            "vov": VectorOfVectors([[1, 2], [3], [4, 5, 6]]),
        }
    )
    with pytest.raises(NotImplementedError, match="vov"):
        take_table_rows(tbl, np.array([0, 1]))
