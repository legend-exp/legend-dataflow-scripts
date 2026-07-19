from __future__ import annotations

from lgdo import Array, Table, WaveformTable

__all__ = ["take_table_rows"]


def take_table_rows(tbl: Table, idx) -> Table:
    """Build a new :class:`lgdo.Table` from a row subset of *tbl*.

    Row-identical to re-reading the same rows from disk, so it can replace a
    second ``lh5.read`` of data that is already in memory.

    Parameters
    ----------
    tbl
        Input table.  Supported column types are :class:`lgdo.WaveformTable`
        (with equal-sized ``values``) and :class:`lgdo.Array` (including
        ``ArrayOfEqualSizedArrays``).
    idx
        Integer index array or boolean mask selecting the rows to keep.

    Returns
    -------
    subset
        New table holding copies of the selected rows; ``units`` attributes
        are preserved.

    Raises
    ------
    NotImplementedError
        If a column type is not supported (e.g. ``VectorOfVectors``).
    """
    col_dict = {}
    for name in tbl:
        col = tbl[name]
        if isinstance(col, WaveformTable):
            values = col.values
            if not hasattr(values, "nda"):
                msg = (
                    f"take_table_rows does not support waveform values of type "
                    f"{type(values).__name__} (column {name!r})"
                )
                raise NotImplementedError(msg)
            col_dict[name] = WaveformTable(
                t0=col.t0.nda[idx],
                t0_units=col.t0.attrs.get("units"),
                dt=col.dt.nda[idx],
                dt_units=col.dt.attrs.get("units"),
                values=values.nda[idx],
                values_units=values.attrs.get("units"),
            )
        elif isinstance(col, Array):
            attrs = {k: v for k, v in col.attrs.items() if k != "datatype"}
            col_dict[name] = type(col)(nda=col.nda[idx], attrs=attrs)
        else:
            msg = (
                f"take_table_rows does not support column type "
                f"{type(col).__name__} (column {name!r})"
            )
            raise NotImplementedError(msg)
    return Table(col_dict=col_dict)
