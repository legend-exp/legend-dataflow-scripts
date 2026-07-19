from __future__ import annotations

import argparse
import copy
import pickle as pkl
from pathlib import Path

import lh5
import numpy as np
from dbetto.catalog import Props
from dspeed import build_dsp
from pygama.pargen.data_cleaning import get_cut_indexes
from pygama.pargen.pz_correct import PZCorrect

from ....utils import (
    build_log,
    check_pulser_mask,
    convert_dict_np_to_float,
    expand_filelist,
    get_is_recovering_mask,
    get_pulser_mask,
    prepare_output_paths,
    require_config_keys,
    take_table_rows,
)


def par_geds_dsp_pz() -> None:
    """Determine the pole-zero (PZ) decay constant(s) for HPGe waveforms.

    CLI entry point registered as ``par-geds-dsp-pz``.  Selects high-energy
    events from raw LH5 files (above an ADC threshold, excluding pulser and
    discharge-recovery events), processes them through a preliminary DSP chain,
    and uses :class:`pygama.pargen.pz_correct.PZCorrect` to fit either a
    single- or double-exponential decay model to the waveform tails.

    The fitted time constant(s) are written to *output-file* in JSON/YAML
    format.  Optionally, diagnostic waveform and slope plots are serialised to
    a pickle file at *plot-path*.

    Notes
    -----
    **Command-line arguments**

    ``--processing-chain`` : list of str
        Processing chain configuration file(s).
    ``--config-file`` : list of str
        PZ calibration configuration file(s).  Must contain ``run_tau``
        (bool), ``threshold`` (ADC), ``n_events`` (int), ``wf_field`` (str),
        and ``mode`` (``"single"`` or ``"double"``).
    ``--log-config`` : str, optional
        Logging configuration file.
    ``--raw-table-name`` : str
        LH5 table path within the raw file (e.g. ``ch1057600/raw``).
    ``--plot-path`` : str, optional
        Output path for the diagnostic pickle file.
    ``--output-file`` : str
        Output path for the fitted PZ parameters (JSON/YAML).
    ``--pulser-file`` : str, optional
        Path to the pulser mask file.
    ``-p`` / ``--no-pulse``
        Flag indicating that no pulser is present; skips pulser masking.
    ``--raw-files`` : list of str
        Raw LH5 input file(s).
    ``--pz-files`` : list of str, optional
        Alternative input file(s) used instead of ``--raw-files`` when
        present (e.g. preselected PZ calibration events).
    ``--log`` : str, optional
        Path to the log file.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--log", help="log file", type=str)
    argparser.add_argument(
        "-p", "--no-pulse", help="no pulser present", action="store_true"
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
    argparser.add_argument(
        "--log-config", help="Log config file", type=str, required=False, default={}
    )

    argparser.add_argument(
        "--raw-table-name", help="raw table name", type=str, required=True
    )

    argparser.add_argument("--plot-path", help="plot path", type=str, required=False)
    argparser.add_argument("--output-file", help="output file", type=str, required=True)

    argparser.add_argument(
        "--pulser-file", help="pulser file", type=str, required=False
    )

    argparser.add_argument("--raw-files", help="input files", nargs="*", type=str)
    argparser.add_argument("--pz-files", help="input files", nargs="*", type=str)
    args = argparser.parse_args()

    log = build_log(args.log_config, args.log)

    prepare_output_paths(args.output_file, args.plot_path)

    kwarg_dict = Props.read_from(args.config_file)
    require_config_keys(kwarg_dict, ["run_tau"], f"pz config ({args.config_file})")

    if kwarg_dict["run_tau"] is True:
        require_config_keys(
            kwarg_dict, ["threshold"], f"pz config ({args.config_file})"
        )
        dsp_config = Props.read_from(args.processing_chain)
        kwarg_dict.pop("run_tau")
        # prefer dedicated pz files when given; an empty pz filelist falls
        # back to the raw files
        input_file = []
        if args.pz_files:
            if len(args.pz_files) == 1 and Path(args.pz_files[0]).suffix == ".filelist":
                with Path(args.pz_files[0]).open() as f:
                    input_file = [line for line in map(str.strip, f) if line]
            else:
                input_file = args.pz_files
        if len(input_file) == 0:
            input_file = expand_filelist(args.raw_files, "--raw-files")

        msg = f"Reading Data for {args.raw_table_name} from:"
        log.debug(msg)
        log.debug(input_file)

        data = lh5.read(
            args.raw_table_name,
            input_file,
            field_mask=["daqenergy", "timestamp", "t_sat_lo"],
        )
        daqenergy = data["daqenergy"].nda
        timestamps = data["timestamp"].nda
        t_sat_lo = data["t_sat_lo"].nda
        threshold = kwarg_dict.pop("threshold")

        if args.no_pulse is False and (
            args.pz_files is None or len(args.pz_files) == 0
        ):
            if args.pulser_file is None:
                msg = "either --pulser-file or --no-pulse is required"
                raise ValueError(msg)
            mask = get_pulser_mask(args.pulser_file)
            check_pulser_mask(mask, data, args.raw_table_name)
        else:
            mask = np.full(len(data), False)

        discharges = t_sat_lo > 0
        discharge_timestamps = np.where(timestamps[discharges])[0]
        is_recovering = get_is_recovering_mask(timestamps, discharge_timestamps)
        cuts = np.where((daqenergy > threshold) & (~mask) & (~is_recovering))[0]
        msg = f"{len(cuts)} events passed threshold and pulser cuts"
        log.debug(msg)
        log.debug(cuts)
        del data, daqenergy, timestamps, t_sat_lo, mask, is_recovering, discharges
        del discharge_timestamps
        tb_data = lh5.read(
            args.raw_table_name,
            input_file,
            idx=cuts,
            n_rows=kwarg_dict["n_events"] * 2,
        )

        dsp_config_optimise_removed = copy.deepcopy(dsp_config)
        if "tau1" in dsp_config["outputs"]:
            dsp_config_optimise_removed["outputs"].remove("tau1")
        if "tau2" in dsp_config["outputs"]:
            dsp_config_optimise_removed["outputs"].remove("tau2")
        if "frac" in dsp_config["outputs"]:
            dsp_config_optimise_removed["outputs"].remove("frac")

        tb_out = build_dsp(raw_in=tb_data, dsp_config=dsp_config_optimise_removed)
        log.debug("Processed Data")
        cut_parameters = kwarg_dict.get("cut_parameters", None)
        if cut_parameters is not None:
            idxs = get_cut_indexes(tb_out, cut_parameters=cut_parameters)
            log.debug("Applied cuts")
            msg = f"{len(idxs)} events passed cuts"
            log.debug(msg)
            # tb_data already holds exactly the rows cuts[:2*n_events], so
            # subset it in memory instead of re-reading the raw files
            # (row-identical: the read idx was cuts[:2*n_events][idxs] with
            # the first n_events kept, both in ascending row order)
            del tb_out
            tb_data = take_table_rows(
                tb_data, np.where(idxs)[0][: kwarg_dict.pop("n_events")]
            )

        tau = PZCorrect(
            dsp_config,
            kwarg_dict["wf_field"],
            debug_mode=kwarg_dict.get("debug_mode", False),
        )
        log.debug("Calculating pz constant")
        if kwarg_dict["mode"] == "single":
            tau.get_single_decay_constant(
                tb_data, kwarg_dict.get("slope_param", "tail_slope")
            )
            msg = f"Found tau: {tau.output_dict['pz']['tau1']}+- {tau.output_dict['pz']['tau1_err']}"
            log.debug(msg)
        elif kwarg_dict["mode"] == "double":
            tau.get_dpz_decay_constants(
                tb_data,
                kwarg_dict.get("percent_tau1_fit", 0.1),
                kwarg_dict.get("percent_tau2_fit", 0.2),
                kwarg_dict.get("offset_from_wf_max", 10),
                kwarg_dict.get("superpulse_bl_idx", 25),
                kwarg_dict.get("superpulse_window_width", 13),
            )
            log.debug("found dpz constants : ")
            for entry in ["tau1", "tau2", "frac"]:
                msg = f"{entry}:{tau.output_dict['pz'][entry]}+- {tau.output_dict['pz'][f'{entry}_err']}"
                log.debug(msg)
        else:
            msg = f"Unknown mode: {kwarg_dict['mode']}, must be either single or double"
            raise ValueError(msg)
        tau.dsp_config = dsp_config_optimise_removed

        if args.plot_path:
            plot_dict = tau.plot_waveforms_after_correction(
                tb_data,
                kwarg_dict.get("wf_pz_field", "wf_pz"),
                norm_param=kwarg_dict.get("norm_param", "pz_mean"),
                xlim=[0, len(tb_data[kwarg_dict["wf_field"]]["values"].nda[0])],
            )

            zoomed = tau.plot_waveforms_after_correction(
                tb_data,
                kwarg_dict.get("wf_pz_field", "wf_pz"),
                norm_param=kwarg_dict.get("norm_param", "pz_mean"),
                xlim=[400, len(tb_data[kwarg_dict["wf_field"]]["values"].nda[0])],
                ylim=[0.8, 1.1],
            )

            plot_dict.update({"waveforms_zoomed": zoomed["waveforms"]})

            plot_dict.update(
                tau.plot_slopes(
                    tb_data, kwarg_dict.get("final_slope_param", "pz_slope")
                )
            )
            plot_dict.update(
                tau.plot_slopes(
                    tb_data, kwarg_dict.get("final_slope_param", "pz_slope"), True
                )
            )

            with Path(args.plot_path).open("wb") as f:
                pkl.dump({"pz": plot_dict}, f, protocol=pkl.HIGHEST_PROTOCOL)
        out_dict = convert_dict_np_to_float(tau.output_dict)
    else:
        out_dict = {}

    Props.write_to(args.output_file, out_dict)
