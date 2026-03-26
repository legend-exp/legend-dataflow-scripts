from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from dbetto.catalog import Props
from lgdo import lh5
from pygama.flow.file_db import FileDB


def build_filedb() -> None:
    """Build a :class:`pygama.flow.file_db.FileDB` from a directory scan.

    CLI entry point registered as ``build-filedb``.  Scans *scan-path* for raw
    LH5 files, inspects their table/column structure, extracts the earliest
    UNIX timestamp found in each file, validates timestamp sanity, drops any
    files whose path matches an entry in the ignore-keys list, and writes the
    resulting ``FileDB`` to disk.

    The ``first_timestamp`` column added to the database is used downstream by
    Snakemake to map file names to calibration validity intervals.

    Notes
    -----
    **Command-line arguments**

    ``--config`` : str
        Path to the FileDB configuration file (JSON/YAML).
    ``--scan-path`` : str
        Root directory to scan for raw LH5 files.
    ``--output`` : str
        Path at which to write the serialised ``FileDB``.
    ``--ignore-keys`` : str, optional
        Path to a JSON/YAML file containing an ``unprocessable`` list of
        substrings; any file whose path contains a listed substring is excluded.
    ``--log`` : str, optional
        Path to the log file.
    ``--assume-nonsparse`` : flag
        When set, only the first channel table is read per file (speeds up
        scanning of non-sparse files).

    Raises
    ------
    RuntimeError
        If a file scan fails or if no valid timestamp can be found in a file.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", required=True)
    argparser.add_argument("--scan-path", required=True)
    argparser.add_argument("--output", required=True)
    argparser.add_argument("--ignore-keys", required=False)
    argparser.add_argument("--log")
    argparser.add_argument("--assume-nonsparse", action="store_true")
    args = argparser.parse_args()

    config = Props.read_from(args.config)

    if args.log is not None:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.DEBUG, filename=args.log, filemode="w")
    else:
        logging.basicConfig(level=logging.DEBUG)

    logging.getLogger("legendmeta").setLevel(logging.INFO)
    logging.getLogger("numba").setLevel(logging.INFO)
    logging.getLogger("parse").setLevel(logging.INFO)
    logging.getLogger("lgdo").setLevel(logging.INFO)
    logging.getLogger("h5py._conv").setLevel(logging.INFO)

    log = logging.getLogger(__name__)

    if args.ignore_keys is not None:
        ignore = Props.read_from(args.ignore_keys)["unprocessable"]
    else:
        ignore = []

    fdb = FileDB(config, scan=False)
    try:
        fdb.scan_files([args.scan_path])
    except Exception as e:
        msg = f"error when building {args.output} from {args.scan_path}"
        raise RuntimeError(msg) from e
    fdb.scan_files([args.scan_path])
    fdb.scan_tables_columns(dir_files_conform=True)

    # augment dataframe with earliest timestamp found in file

    default = np.finfo("float64").max
    timestamps = np.zeros(len(fdb.df), dtype="float64")

    drop_rows = []
    for i, row in enumerate(fdb.df.itertuples()):
        if any(key in row.raw_file for key in ignore):
            drop_rows.append(i)
            continue

        store = lh5.LH5Store(
            base_path=f"{fdb.data_dir}/{fdb.tier_dirs['raw']}", keep_open=True
        )

        # list of first timestamps for each channel
        loc_timestamps = np.full(
            len(row.raw_tables), fill_value=default, dtype="float64"
        )

        msg = f"finding first timestamp in {fdb.data_dir}/{fdb.tier_dirs['raw']}/{row.raw_file}"
        log.info(msg)

        found = False
        for j, table in enumerate(row.raw_tables):
            try:
                loc_timestamps[j] = store.read(
                    fdb.table_format["raw"].format(ch=table) + "/timestamp",
                    row.raw_file.strip("/"),
                    n_rows=1,
                )[0]
                found = True
            except KeyError:
                pass

            if found and args.assume_nonsparse:
                break

        if (
            (loc_timestamps == default).all() or not found
        ) and row.raw_file not in ignore:
            msg = "something went wrong! no valid first timestamp found. Likely: the file {row.raw_file} is empty"
            raise RuntimeError(msg)

        timestamps[i] = np.min(loc_timestamps)

        msg = f"found {timestamps[i]}"
        log.info(msg)

        if (
            timestamps[i] < 0 or timestamps[i] > 4102444800
        ) and row.raw_file not in ignore:
            msg = f"something went wrong! timestamp {timestamps[i]} does not make sense in {row.raw_file}"
            raise RuntimeError(msg)

    fdb.df["first_timestamp"] = timestamps

    fdb.df = fdb.df.drop(drop_rows)

    fdb.to_disk(args.output, wo_mode="of")
