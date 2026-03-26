from __future__ import annotations

import argparse
from pathlib import Path

from dbetto.catalog import Props


def par_geds_dsp_svm() -> None:
    """Register a pre-trained SVM model file in the DSP parameter database.

    CLI entry point registered as ``par-geds-dsp-svm``.  Reads an existing
    DSP parameter file from *input-file*, appends a ``svm`` key that points to
    the SVM model pickle file via a ``loadlh5``-compatible path template, and
    writes the result to *output-file*.

    The model path is stored as a relative reference using the ``$_``
    directory variable so the parameter file remains portable when the
    production directory is relocated.

    Notes
    -----
    **Command-line arguments**

    ``--log`` : str, optional
        Path to the log file.
    ``--output-file`` : str
        Output path for the updated DSP parameter file (JSON/YAML).
    ``--input-file`` : str
        Input DSP parameter file (JSON/YAML) to update.
    ``--svm-file`` : str
        Path to the serialised SVM model pickle file.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--log", help="log file", type=str)
    argparser.add_argument(
        "--output-file", help="output par file", type=str, required=True
    )
    argparser.add_argument(
        "--input-file", help="input par file", type=str, required=True
    )
    argparser.add_argument("--svm-file", help="svm file", required=True)
    args = argparser.parse_args()

    par_data = Props.read_from(args.input_file)

    file = f"'$_/{Path(args.svm_file).name}'"

    par_data["svm"] = {"model_file": file}

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    Props.write_to(args.output_file, par_data)
