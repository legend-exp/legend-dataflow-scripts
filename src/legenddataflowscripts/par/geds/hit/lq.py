from __future__ import annotations

import argparse
import copy
import inspect
import logging
import pickle as pkl
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dbetto.catalog import Props
from pygama.math.distributions import gaussian
from pygama.pargen.AoE_cal import *  # noqa: F403
from pygama.pargen.lq_cal import *  # noqa: F403
from pygama.pargen.lq_cal import LQCal
from pygama.pargen.utils import load_data

from ....utils import (
    build_log,
    check_pulser_mask,
    convert_dict_np_to_float,
    expand_filelist,
    fill_plot_dict,
    get_pulser_mask,
    prepare_output_paths,
    require_config_keys,
)

log = logging.getLogger(__name__)
warnings.filterwarnings(action="ignore", category=RuntimeWarning)


def get_results_dict(lq_class):
    """Extract serialisable results from a calibrated :class:`~pygama.pargen.lq_cal.LQCal` object.

    Parameters
    ----------
    lq_class : pygama.pargen.lq_cal.LQCal
        Calibrated LQ object after calling :meth:`~pygama.pargen.lq_cal.LQCal.calibrate`.

    Returns
    -------
    dict
        Results dictionary containing the calibration energy parameter, DEP
        mean values per run, drift-time correction fit parameters, cut fit
        parameters, cut value, and survival fractions.
    """
    return {
        "cal_energy_param": lq_class.cal_energy_param,
        "DEP_means": lq_class.timecorr_df.to_dict("index"),
        "rt_correction": lq_class.dt_fit_pars,
        "cut_fit_pars": lq_class.cut_fit_pars.to_dict(),
        "cut_value": lq_class.cut_val,
        "sfs": lq_class.low_side_sf.to_dict("index"),
    }


def lq_calibration(
    data: pd.DataFrame,
    cal_dicts: dict,
    energy_param: str,
    cal_energy_param: str,
    dt_param: str,
    eres_func: callable,
    cdf: callable = gaussian,
    lq_param: str = "lq80",
    selection_string: str = "",
    plot_options: dict | None = None,
    debug_mode: bool = False,
    use_log_pdf: bool = False,
    suffix: str | None = None,
):
    """Calibrate the LQ (late-charge) pulse-shape discriminant.

    Constructs a :class:`~pygama.pargen.lq_cal.LQCal` instance, computes the
    energy-normalised LQ observable ``LQ_Ecorr = lq_param / energy_param``, and
    calls :meth:`~pygama.pargen.lq_cal.LQCal.calibrate`.  The resulting cut
    expressions are appended to *cal_dicts*.

    Parameters
    ----------
    data : pandas.DataFrame
        Event-level data containing *lq_param*, *energy_param*,
        *cal_energy_param*, *dt_param*, and any selection columns.
    cal_dicts : dict
        Mapping of run timestamps → hit-level calibration operations.
        Updated in-place with the LQ cut expression.
    energy_param : str
        Raw (uncalibrated) energy parameter name used for LQ normalisation.
    cal_energy_param : str
        Calibrated energy parameter name.
    dt_param : str
        Drift-time parameter name used for the energy-dependent LQ correction.
    eres_func : callable
        Energy resolution function ``sigma(E)`` used to set cut windows.
    cdf : callable
        Cumulative distribution function used for binned LQ fitting.
        Defaults to :func:`pygama.math.distributions.gaussian`.
    lq_param : str
        Raw LQ parameter name to calibrate.  Defaults to ``"lq80"``.
    selection_string : str
        Pandas query string applied before calibration.  Defaults to ``""``.
    plot_options : dict, optional
        Mapping of ``{label: {"function": callable, "options": dict | None}}``
        passed to :func:`~legenddataflowscripts.utils.fill_plot_dict`.
    debug_mode : bool
        Activates additional diagnostic output.  Defaults to ``False``.
    use_log_pdf : bool
        Build the survival-fraction unbinned NLL fits from the models'
        log-densities (iminuit ``log=True`` mode) — faster, results differ
        at machine-precision level.  Silently ignored when the installed
        pygama does not support it.
    suffix : str, optional
        Appended (underscore-separated) to all output parameter names, e.g.
        ``suffix="foo"`` produces ``LQ_Ecorr_foo`` … ``LQ_Cut_foo``.  Lets
        several LQ parameters be calibrated side by side without their
        hit-level operations colliding.  Requires a pygama version whose
        :meth:`~pygama.pargen.lq_cal.LQCal.calibrate` supports ``suffix``.

    Returns
    -------
    cal_dicts : dict
        Updated calibration operations mapping.
    results_dict : dict
        LQ calibration results from :func:`get_results_dict`.
    plot_dict : dict
        Diagnostic figures (empty when *plot_options* is ``None``).
    lq : pygama.pargen.lq_cal.LQCal
        Calibrated LQ object.
    """

    if use_log_pdf and "use_log_pdf" not in inspect.signature(LQCal).parameters:
        log.warning("installed pygama does not support use_log_pdf, ignoring")
        use_log_pdf = False

    lq = LQCal(
        cal_dicts,
        cal_energy_param,
        dt_param,
        eres_func,
        cdf,
        selection_string,
        debug_mode=debug_mode,
        **({"use_log_pdf": True} if use_log_pdf else {}),
    )

    initial_param = "LQ_Ecorr" if suffix is None else f"LQ_Ecorr_{suffix}"
    data[initial_param] = np.divide(data[lq_param], data[energy_param])

    lq.update_cal_dicts(
        {
            initial_param: {
                "expression": f"{lq_param}/{energy_param}",
                "parameters": {},
            }
        }
    )

    lq.calibrate(data, initial_param, **({} if suffix is None else {"suffix": suffix}))
    return cal_dicts, get_results_dict(lq), fill_plot_dict(lq, data, plot_options), lq


def run_lq_calibration(
    data,
    cal_dicts,
    results_dicts,
    object_dicts,
    plot_dicts,
    configs,
    debug_mode=False,
    # gen_plots=True,
):
    """Run the LQ calibration and update all output dictionaries.

    Wraps :func:`lq_calibration` to operate on timestamp-keyed dictionaries
    and merge LQ results into the shared output structures used by the
    dataflow.

    Parameters
    ----------
    data : pandas.DataFrame
        Event-level data with the LQ, energy, drift-time, and cut columns.
    cal_dicts : dict
        ``{timestamp: operations_dict}`` mapping of existing hit-level
        calibration operations.  Updated with the LQ cut expression.
    results_dicts : dict
        ``{timestamp: results_dict}`` mapping of preceding calibration results
        (energy calibration and partition calibration).
    object_dicts : dict
        ``{timestamp: object_dict}`` mapping of pickled calibration objects.
    plot_dicts : dict
        ``{timestamp: plot_dict}`` mapping of existing diagnostic figures.
    configs : dict or str or list
        LQ calibration configuration.  Must contain ``run_lq`` (bool),
        ``cal_energy_param``, ``cut_field``, and ``params`` — a mapping of
        ``{name: param_config}`` with one entry per LQ parameter to
        calibrate.  Each *param_config* must contain ``lq_param`` and
        ``energy_param``; optional keys are ``dt_param`` (default
        ``"dt_eff"``), ``cdf``, ``suffix`` (appended to the output parameter
        names so entries don't collide), and ``plot_options``.  Optional
        top-level key ``use_log_pdf`` (bool, default false): build the
        survival-fraction unbinned fits from the models' log-densities
        (iminuit ``log=True``) — substantially faster, results differ at
        machine-precision level; requires a pygama version with
        ``use_log_pdf`` support (silently ignored otherwise).
    debug_mode : bool
        Activates additional diagnostic output.  Defaults to ``False``.

    Returns
    -------
    cal_dicts : dict
        Updated calibration operations mappings.
    out_result_dicts : dict
        Updated results mappings; LQ results are nested per ``params`` entry
        under the ``lq`` key.
    out_object_dicts : dict
        Updated object mappings including one ``LQCal`` instance per
        ``params`` entry.
    out_plot_dicts : dict
        Updated plot mappings including LQ diagnostic figures per ``params``
        entry.
    """
    if isinstance(configs, str | list):
        configs = Props.read_from(configs)

    require_config_keys(configs, ["run_lq"], "lq calibration config")

    if configs["run_lq"] is True:
        require_config_keys(
            configs,
            ["cal_energy_param", "cut_field", "params"],
            "lq calibration config",
        )
        lq_objs = {}
        lq_plot_dict = {}
        out_dicts = {}
        try:
            eres = copy.deepcopy(
                results_dicts[next(iter(results_dicts))]["partition_ecal"][
                    configs["cal_energy_param"]
                ]["eres_linear"]
            )

            def eres_func(x):
                return eval(eres["expression"], dict(x=x, **eres["parameters"]))

            if np.isnan(eres_func(2000)):
                raise RuntimeError
        except (KeyError, RuntimeError):
            try:
                eres = copy.deepcopy(
                    results_dicts[next(iter(results_dicts))]["ecal"][
                        configs["cal_energy_param"]
                    ]["eres_linear"]
                )

                def eres_func(x):
                    return eval(eres["expression"], dict(x=x, **eres["parameters"]))

            except KeyError:

                def eres_func(x):
                    return x * np.nan

        for name, param_config in configs["params"].items():
            require_config_keys(
                param_config,
                ["lq_param", "energy_param"],
                f"lq calibration config params entry '{name}'",
            )
            if "plot_options" in param_config:
                for field, item in param_config["plot_options"].items():
                    param_config["plot_options"][field]["function"] = eval(
                        item["function"]
                    )

            msg = f"starting lq calibration for {name}"
            log.info(msg)
            start = time.time()
            cal_dicts, out_dict, lq_plots, lq_obj = lq_calibration(
                data,
                cal_dicts=cal_dicts,
                energy_param=param_config["energy_param"],
                cal_energy_param=configs["cal_energy_param"],
                dt_param=param_config.get("dt_param", "dt_eff"),
                eres_func=eres_func,
                cdf=eval(param_config.get("cdf", "gaussian")),
                lq_param=param_config["lq_param"],
                selection_string=f"{configs['cut_field']}&(~is_pulser)",
                plot_options=param_config.get("plot_options", None),
                debug_mode=debug_mode | configs.get("debug_mode", False),
                use_log_pdf=configs.get("use_log_pdf", False),
                suffix=param_config.get("suffix", None),
            )
            msg = f"lq calibration for {name} took {time.time() - start:.2f} seconds"
            log.info(msg)
            # need to change eres func as can't pickle lambdas
            try:
                lq_obj.eres_func = results_dicts[next(iter(results_dicts))][
                    "partition_ecal"
                ][configs["cal_energy_param"]]["eres_linear"]
            except KeyError:
                lq_obj.eres_func = {}
            out_dicts[name] = out_dict
            lq_objs[name] = copy.deepcopy(lq_obj)
            lq_plot_dict[name] = copy.deepcopy(lq_plots)
    else:
        out_dicts = {}
        lq_objs = {}
        lq_plot_dict = {}

    out_result_dicts = {}
    for tstamp, result_dict in results_dicts.items():
        out_result_dicts[tstamp] = dict(**result_dict, lq=dict(out_dicts))

    out_object_dicts = {}
    for tstamp, object_dict in object_dicts.items():
        out_object_dicts[tstamp] = dict(**object_dict, lq=lq_objs)

    common_dict = lq_plot_dict.pop("common") if "common" in list(lq_plot_dict) else None
    out_plot_dicts = {}
    for tstamp, plot_dict in plot_dicts.items():
        if "common" in list(plot_dict) and common_dict is not None:
            plot_dict["common"].update(common_dict)
        elif common_dict is not None:
            plot_dict["common"] = common_dict
        plot_dict.update({"lq": lq_plot_dict})
        out_plot_dicts[tstamp] = plot_dict

    return cal_dicts, out_result_dicts, out_object_dicts, out_plot_dicts


def par_geds_hit_lq() -> None:
    """Calibrate the LQ pulse-shape discriminant and write hit-level parameters.

    CLI entry point registered as ``par-geds-hit-lq``.  Loads DSP-level data
    for a single detector channel, applies energy threshold and pulser masks,
    and runs :func:`run_lq_calibration` to derive the energy-normalised LQ
    observable, its drift-time correction, and the DEP-based cut value.

    Results are written to *hit-pars* (JSON/YAML) and the calibration objects
    are serialised to *lq-results* (pickle).

    Notes
    -----
    **Command-line arguments**

    ``files`` : list of str
        One or more file lists (``.filelist``) containing DSP LH5 paths.
    ``--pulser-file`` : str, optional
        Path to the pulser mask file.
    ``--tcm-filelist`` : str, optional
        Unused placeholder.
    ``--ecal-file`` : str
        Energy calibration output file (JSON/YAML with ``pars`` and
        ``results`` keys).
    ``--eres-file`` : str
        Energy calibration pickle file containing calibration objects.
    ``--inplots`` : str, optional
        Existing pickle plot file to merge with LQ plots.
    ``--log`` : str, optional
        Path to the log file.
    ``--log-config`` : str, optional
        Logging configuration file.
    ``--config-file`` : list of str
        LQ calibration configuration file(s).  Must contain ``run_lq``
        (bool), ``cal_energy_param``, ``cut_field``, ``threshold``, and
        ``params`` — one entry per LQ parameter to calibrate (see
        :func:`run_lq_calibration`).
    ``--table-name`` : str
        LH5 table path within the DSP files.
    ``--timestamp`` : str
        Run timestamp label.  Defaults to ``"20000101T000000Z"``.
    ``--plot-file`` : str, optional
        Output path for diagnostic plots (pickle).
    ``--hit-pars`` : str
        Output path for the LQ hit parameters (JSON/YAML).
    ``--lq-results`` : str
        Output path for the serialised LQ calibration object (pickle).
    ``-d`` / ``--debug``
        Enable debug mode for additional diagnostic output.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("files", help="files", nargs="*", type=str)
    argparser.add_argument(
        "--pulser-file", help="pulser_file", type=str, required=False
    )
    argparser.add_argument(
        "--tcm-filelist", help="tcm_filelist", type=str, required=False
    )

    argparser.add_argument("--ecal-file", help="ecal_file", type=str, required=True)
    argparser.add_argument("--eres-file", help="eres_file", type=str, required=True)
    argparser.add_argument("--inplots", help="in_plot_path", type=str, required=False)

    argparser.add_argument("--log", help="log_file", type=str)
    argparser.add_argument(
        "--log-config", help="Log config file", type=str, required=False, default={}
    )

    argparser.add_argument(
        "--config-file", help="Config file", type=str, nargs="*", required=True
    )

    argparser.add_argument("--table-name", help="table name", type=str, required=True)
    argparser.add_argument(
        "--timestamp",
        help="timestamp",
        type=str,
        required=False,
        default="20000101T000000Z",
    )

    argparser.add_argument("--plot-file", help="plot_file", type=str, required=False)
    argparser.add_argument("--hit-pars", help="hit_pars", type=str)
    argparser.add_argument("--lq-results", help="lq_results", type=str)

    argparser.add_argument("-d", "--debug", help="debug_mode", action="store_true")
    args = argparser.parse_args()

    build_log(args.log_config, args.log)

    prepare_output_paths(args.plot_file, args.hit_pars, args.lq_results)

    kwarg_dict = Props.read_from(args.config_file)
    require_config_keys(kwarg_dict, ["run_lq"], f"lq config ({args.config_file})")

    ecal_dict = Props.read_from(args.ecal_file)
    cal_dict = ecal_dict["pars"]["operations"]
    eres_dict = ecal_dict["results"]

    if args.inplots:
        with Path(args.inplots).open("rb") as r:
            plot_dict = pkl.load(r)
    else:
        plot_dict = {}

    with Path(args.eres_file).open("rb") as o:
        object_dict = pkl.load(o)

    if kwarg_dict["run_lq"] is True:
        require_config_keys(
            kwarg_dict,
            ["cal_energy_param", "cut_field", "threshold", "params"],
            f"lq config ({args.config_file})",
        )
        files = expand_filelist(args.files)

        params = [
            kwarg_dict["cal_energy_param"],
            kwarg_dict["cut_field"],
        ]
        for param_config in kwarg_dict["params"].values():
            params.append(param_config["lq_param"])
            params.append(param_config["energy_param"])
            params.append(param_config.get("dt_param", "dt_eff"))
        params = list(dict.fromkeys(params))

        # load data in
        data, threshold_mask = load_data(
            files,
            args.table_name,
            cal_dict,
            params=params,
            threshold=kwarg_dict["threshold"],
            return_selection_mask=True,
        )

        msg = f"Loaded {len(data)} events"
        log.info(msg)

        if args.pulser_file is not None:
            mask = get_pulser_mask(
                pulser_file=args.pulser_file,
            )
        else:
            mask = np.zeros(len(threshold_mask), dtype=bool)

        check_pulser_mask(mask, threshold_mask, args.table_name)
        data["is_pulser"] = mask[threshold_mask]

        msg = f"{len(data.query('~is_pulser'))}  non pulser events"
        log.info(msg)

        data["run_timestamp"] = args.timestamp

        out_dicts, results_dicts, lq_dict, plot_dicts = run_lq_calibration(
            data,
            cal_dicts={args.timestamp: cal_dict},
            results_dicts={args.timestamp: eres_dict},
            object_dicts={args.timestamp: object_dict},
            plot_dicts={args.timestamp: plot_dict},
            configs=kwarg_dict,
            debug_mode=args.debug,
        )
        cal_dict = out_dicts[args.timestamp]
        results_dict = results_dicts[args.timestamp]
        plot_dict = plot_dicts[args.timestamp]
        lq = lq_dict[args.timestamp]["lq"]

    else:
        lq = {}
        results_dict = {}

    if args.plot_file:
        with Path(args.plot_file).open("wb") as w:
            pkl.dump(plot_dict, w, protocol=pkl.HIGHEST_PROTOCOL)

    final_hit_dict = convert_dict_np_to_float(
        {
            "pars": {"operations": cal_dict},
            "results": results_dict,
        }
    )
    Props.write_to(args.hit_pars, final_hit_dict)

    final_object_dict = dict(
        **object_dict,
        lq=lq,
    )
    with Path(args.lq_results).open("wb") as w:
        pkl.dump(final_object_dict, w, protocol=pkl.HIGHEST_PROTOCOL)
