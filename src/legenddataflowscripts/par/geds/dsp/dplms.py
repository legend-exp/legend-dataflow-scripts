from __future__ import annotations

import argparse
import pickle as pkl
import time
from pathlib import Path

import numpy as np
import pygama.math.distributions as pmd  # noqa: F401
from dbetto.catalog import Props
from lgdo import Array, Table, lh5
from pygama.pargen.data_cleaning import generate_cuts
from pygama.pargen.dplms_ge_dict import dplms_ge_dict
from pygama.pargen.dsp_optimize import run_one_dsp

from ....utils import build_log, convert_dict_np_to_float


def par_geds_dsp_dplms() -> None:
    """Compute DPLMS optimal filter coefficients for HPGe detectors.

    CLI entry point registered as ``par-geds-dsp-dplms``.  The *Discrete
    Prolate Laplacian Maximum Signal* (DPLMS) filter is constructed from two
    data samples:

    1. **FFT baselines** - low-energy (``daqenergy <= 10`` ADC) events from
       dedicated FFT run files used to characterise the noise power spectrum.
    2. **Calibration peak events** - gamma-line events read from the
       *peak-file* produced by :func:`par_geds_dsp_evtsel`.

    The coefficients are computed by
    :func:`pygama.pargen.dplms_ge_dict.dplms_ge_dict` and written to an LH5
    file at *lh5-path* (so that the DSP chain can load them with
    ``loadlh5(...)``).  The DSP parameter database is updated with the filter
    configuration and written to *dsp-pars*.

    Notes
    -----
    **Command-line arguments**

    ``--fft-raw-filelist`` : str
        Path to a text file listing the FFT raw LH5 input files.
    ``--peak-file`` : str
        LH5 file containing preselected calibration peak events.
    ``--inplots`` : str, optional
        Existing pickle plot file to update with DPLMS plots.
    ``--database`` : str
        Path to the existing DSP parameter database (JSON/YAML).
    ``--log`` : str, optional
        Path to the log file.
    ``--log-config`` : str, optional
        Logging configuration file.
    ``--processing-chain`` : list of str
        Processing chain configuration file(s).
    ``--config-file`` : list of str
        DPLMS configuration file(s).  Must contain ``run_dplms`` (bool),
        ``n_baselines`` (int), and ``peaks_kev`` (list).
    ``--channel`` : str
        Channel identifier; used as the HDF5 group name in *lh5-path*.
    ``--raw-table-name`` : str
        LH5 table path within the raw and peak files.
    ``--dsp-pars`` : str
        Output path for the updated DSP parameter database (JSON/YAML).
    ``--lh5-path`` : str
        Output LH5 file path for the DPLMS filter coefficients.
    ``--plot-path`` : str, optional
        Output path for diagnostic plots (pickle).
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--fft-raw-filelist", help="fft_raw_filelist", type=str)
    argparser.add_argument("--peak-file", help="tcm_filelist", type=str, required=True)
    argparser.add_argument("--inplots", help="in_plot_path", type=str)
    argparser.add_argument("--database", help="database", type=str, required=True)

    argparser.add_argument("--log", help="log_file", type=str)
    argparser.add_argument(
        "--log-config", help="Log config file", type=str, required=False, default={}
    )

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

    argparser.add_argument("--channel", help="channel", type=str, required=True)
    argparser.add_argument(
        "--raw-table-name", help="raw table name", type=str, required=True
    )

    argparser.add_argument("--dsp-pars", help="dsp_pars", type=str, required=True)
    argparser.add_argument("--lh5-path", help="lh5_path", type=str, required=True)
    argparser.add_argument("--plot-path", help="plot_path", type=str)

    args = argparser.parse_args()

    dsp_config = Props.read_from(args.processing_chain)
    log = build_log(args.log_config, args.log)

    t0 = time.time()

    dplms_dict = Props.read_from(args.config_file)
    db_dict = Props.read_from(args.database)

    if dplms_dict["run_dplms"] is True:
        with Path(args.fft_raw_filelist).open() as f:
            fft_files = sorted(f.read().splitlines())

        t0 = time.time()
        log.info("\nLoad fft data")
        energies = lh5.read_as(
            f"{args.raw_table_name}/daqenergy", fft_files, library="np"
        )
        eidxs = np.where(energies <= 10)[0]
        raw_fft = lh5.read(
            args.raw_table_name,
            fft_files,
            n_rows=dplms_dict["n_baselines"],
            idx=eidxs,
        )
        t1 = time.time()
        msg = f"Time to load fft data {(t1 - t0):.2f} s, total events {len(raw_fft)}"
        log.info(msg)

        log.info("\nRunning event selection")
        peaks_kev = np.array(dplms_dict["peaks_kev"])
        # kev_widths = [tuple(kev_width) for kev_width in dplms_dict["kev_widths"]]

        peaks_rounded = [int(peak) for peak in peaks_kev]
        peaks = lh5.read_as(f"{args.raw_table_name}/peak", args.peak_file, library="np")
        ids = np.isin(peaks, peaks_rounded)
        peaks = peaks[ids]
        # idx_list = [np.where(peaks == peak)[0] for peak in peaks_rounded]

        raw_cal = lh5.read(args.raw_table_name, args.peak_file, idx=ids)
        msg = f"Time to run event selection {(time.time() - t1):.2f} s, total events {len(raw_cal)}"
        log.info(msg)

        if isinstance(dsp_config, str | list):
            dsp_config = Props.read_from(dsp_config)

        dsp_fft = run_one_dsp(raw_fft, dsp_config, db_dict=db_dict)

        cut_dict = generate_cuts(dsp_fft, cut_dict=dplms_dict["bls_cut_pars"])
        log.debug("Cuts are %s", cut_dict)
        idxs = np.full(len(dsp_fft), True, dtype=bool)
        for outname, info in cut_dict.items():
            outcol = dsp_fft.eval(info["expression"], info.get("parameters", None))
            dsp_fft.add_column(outname, outcol)
        for cut in cut_dict:
            idxs = dsp_fft[cut].nda & idxs

        selected_eidxs = eidxs[: len(dsp_fft)]
        raw_fft = lh5.read(
            args.raw_table_name,
            fft_files,
            n_rows=dplms_dict["n_baselines"],
            idx=selected_eidxs[idxs],
        )

        log.debug("Applied Cuts")

        out_dict, plot_dict = dplms_ge_dict(
            raw_fft,
            raw_cal,
            dsp_config,
            db_dict,
            dplms_dict,
            fom_func=eval(dplms_dict.get("fom_func", "pmd.gauss_on_step")),
            display=1 if args.plot_path else 0,
        )
        if args.inplots:
            with Path(args.inplots).open("rb") as r:
                inplot_dict = pkl.load(r)
            inplot_dict.update({"dplms": plot_dict})
        else:
            inplot_dict = {"dplms": plot_dict}

        coeffs = out_dict["dplms"].pop("coefficients")
        dplms_pars = Table(col_dict={"coefficients": Array(coeffs)})
        out_dict["dplms"]["coefficients"] = (
            f"loadlh5('{args.lh5_path}', '{args.channel}/dplms/coefficients')"
        )
        msg = f"DPLMS creation finished in {(time.time() - t0) / 60} minutes"
        log.info(msg)
    else:
        out_dict = {}
        dplms_pars = Table(col_dict={"coefficients": Array([])})
        if args.inplots:
            with Path(args.inplots).open("rb") as r:
                inplot_dict = pkl.load(r)
        else:
            inplot_dict = {}

    db_dict.update(out_dict)

    Path(args.lh5_path).parent.mkdir(parents=True, exist_ok=True)
    lh5.write(
        Table(col_dict={"dplms": dplms_pars}),
        name=args.channel,
        lh5_file=args.lh5_path,
        wo_mode="overwrite",
    )
    Path(args.dsp_pars).parent.mkdir(parents=True, exist_ok=True)
    Props.write_to(args.dsp_pars, convert_dict_np_to_float(db_dict))

    if args.plot_path:
        Path(args.plot_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.plot_path).open("wb") as f:
            pkl.dump(inplot_dict, f, protocol=pkl.HIGHEST_PROTOCOL)
