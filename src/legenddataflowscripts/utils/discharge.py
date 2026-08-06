from __future__ import annotations

import numpy as np

__all__ = ["get_is_recovering_mask"]


def get_is_recovering_mask(
    timestamps: np.ndarray,
    discharge_timestamps: np.ndarray,
    window: float = 0.01,
) -> np.ndarray:
    """Mark events that follow a discharge within a recovery window.

    An event at time ``t`` is flagged when any discharge time ``d`` satisfies
    ``0 < t - d < window``.  Results are bit-identical to the original
    ``O(n_discharges * n_events)`` loop: a sorted search collects a slightly
    widened candidate superset of (event, discharge) pairs, and the original
    floating-point conditions are then evaluated exactly on those pairs, in
    ``O((n + m) log m + n_candidates)``.

    Parameters
    ----------
    timestamps
        Event timestamps (any order).
    discharge_timestamps
        Timestamps of discharge events (any order).
    window
        Length of the recovery window in the same units as the timestamps.

    Returns
    -------
    is_recovering
        Boolean array, one entry per event in *timestamps*.
    """
    ts = np.asarray(timestamps)
    ds = np.sort(np.asarray(discharge_timestamps))
    out = np.zeros(ts.shape, dtype=bool)
    if ds.size == 0 or ts.size == 0:
        return out

    # candidate discharges for event t: d < t (exact: t - d > 0 iff d < t)
    # and d >= t - margin, with margin slightly wider than the window so no
    # pair with fl(t - d) < window can be missed by the rounding of t - margin
    margin = window * 1.001
    lo = np.searchsorted(ds, ts - margin, side="left")
    hi = np.searchsorted(ds, ts, side="left")

    counts = hi - lo
    cand = counts > 0
    if not cand.any():
        return out

    ev_rows = np.nonzero(cand)[0]
    ln = counts[ev_rows]
    total = int(ln.sum())
    # flat indices of each candidate pair's discharge in ds
    seg_starts = np.cumsum(ln) - ln
    offsets = np.arange(total) - np.repeat(seg_starts, ln)
    d_idx = np.repeat(lo[ev_rows], ln) + offsets

    # evaluate the original per-pair conditions with the original arithmetic
    diff = ts[np.repeat(ev_rows, ln)] - ds[d_idx]
    hit = (diff < window) & (diff > 0)

    out[ev_rows] = np.logical_or.reduceat(hit, seg_starts)
    return out
