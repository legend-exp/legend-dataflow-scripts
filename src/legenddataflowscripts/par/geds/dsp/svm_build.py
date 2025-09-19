from __future__ import annotations

import argparse
import pickle as pkl
from pathlib import Path

from dbetto.catalog import Props
from lgdo import lh5
from sklearn.svm import SVC

from ....utils import build_log


def par_geds_dsp_svm_build() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--log", help="log file", type=str)

    argparser.add_argument(
        "--log-config", help="Log config file", type=str, required=False, default={}
    )

    argparser.add_argument(
        "--output-file", help="output SVM file", type=str, required=True
    )
    argparser.add_argument(
        "--train-data", help="input data file", nargs="*", default=None
    )
    argparser.add_argument(
        "--train-hyperpars", help="input hyperparameter file", nargs="*", default=None
    )
    args = argparser.parse_args()

    log = build_log(args.log_config, args.log)

    if args.train_data is not None and len(args.train_data) > 0:
        # Load files
        tb = lh5.read("ml_train/dsp", args.train_data)
        log.debug("loaded data")

        hyperpars = Props.read_from(args.train_hyperpars)

        # Define training inputs
        dwts_norm = tb["dwt_norm"].nda
        labels = tb["dc_label"].nda

        log.debug("training model")
        # Initialize and train SVM
        svm = SVC(
            random_state=int(hyperpars["random_state"]),
            kernel=hyperpars["kernel"],
            decision_function_shape=hyperpars["decision_function_shape"],
            class_weight=hyperpars["class_weight"],
            C=float(hyperpars["C"]),
            gamma=float(hyperpars["gamma"]),
            cache_size=1000,
        )

        svm.fit(dwts_norm, labels)
        log.debug("trained model")
    else:
        svm = None

    # Save trained model with pickle
    with Path(args.output_file).open("wb") as svm_file:
        pkl.dump(svm, svm_file, protocol=pkl.HIGHEST_PROTOCOL)
