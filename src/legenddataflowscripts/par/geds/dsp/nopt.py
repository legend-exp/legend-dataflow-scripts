from __future__ import annotations

import argparse
import pickle as pkl
import time
from pathlib import Path

import numpy as np
import pygama.pargen.noise_optimization as pno
from dbetto.catalog import Props
from dspeed import build_dsp
from lgdo import lh5
from pygama.pargen.data_cleaning import generate_cuts, get_cut_indexes

from ....utils import build_log


def par_geds_dsp_nopt() -> None:
    """Optimise DSP filter parameters for noise performance.

    CLI entry point registered as ``par-geds-dsp-nopt``.  Selects low-energy
    baseline events (``daqenergy ≤ 10`` ADC) from raw LH5 files, applies
    quality cuts derived from the preliminary DSP output, and passes the clean
    baseline sample to :func:`pygama.pargen.noise_optimization.noise_optimization`
    to determine optimal filter shaping parameters for noise rejection.

    The resulting noise-optimisation parameters are merged with the existing
    DSP parameter database and written to *dsp-pars*.  Optionally, diagnostic
    plots are serialised to *plot-path*.

    Notes
    -----
    **Command-line arguments**

    ``--raw-filelist`` : str
        Path to a text file listing the raw LH5 input files (one per line).
    ``--database`` : str
        Path to the existing DSP parameter database (JSON/YAML).
    ``--inplots`` : str, optional
        Existing pickle plot file to update with new noise-optimisation plots.
    ``--log`` : str, optional
        Path to the log file.
    ``--processing-chain`` : list of str
        Processing chain configuration file(s).
    ``--config-file`` : list of str
        Noise optimisation configuration file(s).  Must contain ``run_nopt``
        (bool), ``n_events`` (int), and ``cut_pars``.
    ``--log-config`` : str, optional
        Logging configuration file.
    ``--raw-table-name`` : str
        LH5 table path within the raw file.
    ``--dsp-pars`` : str
        Output path for the merged DSP parameter database (JSON/YAML).
    ``--plot-path`` : str, optional
        Output path for diagnostic plots (pickle).
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--raw-filelist", help="raw_filelist", type=str)
    argparser.add_argument("--database", help="database", type=str, required=True)
    argparser.add_argument("--inplots", help="inplots", type=str)

    argparser.add_argument("--log", help="log_file", type=str)

    argparser.add_argument(
        "--processing-chain",
        help="Processing chain config",
        type=str,
        nargs="*",
        required=True,
    )
    argparser.add_argument(
        "--config-file", help="Config file", type=str, nargs="*", required=True
    )
    argparser.add_argument(
        "--log-config", help="Log config file", type=str, required=False, default={}
    )

    argparser.add_argument(
        "--raw-table-name", help="raw table name", type=str, required=True
    )

    argparser.add_argument("--dsp-pars", help="dsp_pars", type=str, required=True)
    argparser.add_argument("--plot-path", help="plot_path", type=str)

    args = argparser.parse_args()

    dsp_config = Props.read_from(args.processing_chain)
    log = build_log(args.log_config, args.log)

    t0 = time.time()

    opt_dict = Props.read_from(args.config_file)
    db_dict = Props.read_from(args.database)

    if opt_dict.pop("run_nopt") is True:
        with Path(args.raw_filelist).open() as f:
            files = f.read().splitlines()

        raw_files = sorted(files)

        energies = lh5.read_as(
            f"{args.raw_table_name}/daqenergy", raw_files, library="np"
        )
        idxs = np.where(energies <= 10)[0]
        tb_data = lh5.read(
            args.raw_table_name, raw_files, n_rows=opt_dict["n_events"], idx=idxs
        )
        t1 = time.time()
        msg = f"Time to open raw files {t1 - t0:.2f} s, n. baselines {len(tb_data)}"
        log.info(msg)

        msg = f"Select baselines {len(tb_data)}"
        log.info(msg)
        dsp_data = build_dsp(raw_in=tb_data, dsp_config=dsp_config)
        cut_dict = generate_cuts(dsp_data, cut_dict=opt_dict.pop("cut_pars"))
        cut_idxs = get_cut_indexes(dsp_data, cut_dict)
        tb_data = lh5.read(
            args.raw_table_name,
            raw_files,
            n_rows=opt_dict.pop("n_events"),
            idx=idxs[cut_idxs],
        )
        msg = f"... {len(tb_data)} baselines after cuts"
        log.info(msg)

        if args.plot_path:
            out_dict, plot_dict = pno.noise_optimization(
                tb_data,
                dsp_config,
                db_dict.copy(),
                opt_dict,
                args.raw_table_name,
                display=1,
            )
        else:
            out_dict = pno.noise_optimization(
                raw_files, dsp_config, db_dict.copy(), opt_dict, args.raw_table_name
            )

        t2 = time.time()
        msg = f"Optimiser finished in {(t2 - t0) / 60} minutes"
        log.info(msg)
    else:
        out_dict = {}
        plot_dict = {}

    if args.plot_path:
        Path(args.plot_path).parent.mkdir(parents=True, exist_ok=True)
        if args.inplots:
            with Path(args.inplots).open("rb") as r:
                old_plot_dict = pkl.load(r)
            plot_dict = dict(noise_optimisation=plot_dict, **old_plot_dict)
        else:
            plot_dict = {"noise_optimisation": plot_dict}
        with Path(args.plot_path).open("wb") as f:
            pkl.dump(plot_dict, f, protocol=pkl.HIGHEST_PROTOCOL)

    Path(args.dsp_pars).parent.mkdir(parents=True, exist_ok=True)
    Props.write_to(args.dsp_pars, dict(nopt_pars=out_dict, **db_dict))
