Developers Guide
================

.. contents::
   :local:
   :depth: 2

Snakemake Rules
---------------

Snakemake is configured around a series of *rules* which specify how to
generate a file or a set of files from a set of input files.  These rules are
defined in the ``Snakefile`` and in files in the ``rules`` directory.

In general, the structure is that a series of rules run on calibration data to
produce a final ``par_{tier}.yaml`` parameter file.  That file is then used by
the tier-building rule to process all physics runs in the tier.

For most calibration steps there are two versions of each rule:

* The **basic version** that processes a single run.
* The **partition version** that groups many runs together before fitting.
  The grouping is defined in ``cal_grouping.yaml`` from the `legend-datasets
  <https://github.com/legend-exp/legend-datasets>`_ repository.

Each rule declares its inputs and outputs together with the shell command or
Python call used to generate them.  Additional parameters can also be defined.
Full details are in the `Snakemake documentation
<https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html>`_.

The calibration scripts are located in :mod:`legenddataflowscripts.par` (one
Python function per script) and the data-tier builders are in
:mod:`legenddataflowscripts.tier`.

Configuration
-------------

The workflow is driven by a ``TextDB``-based configuration directory
(see `dbetto <https://dbetto.readthedocs.io>`_).  Each configuration file is
a JSON or YAML file; validity is expressed through a ``validity.yaml`` catalog.
The active configuration for a given timestamp and data type is retrieved with:

.. code-block:: python

   from dbetto import TextDB

   db = TextDB("/path/to/configs", lazy=True)
   cfg = db.on(timestamp, system=datatype).snakemake_rules[f"tier_{tier}"]

The ``snakemake_rules`` section of the configuration maps tier names to
``options`` (processing settings) and ``inputs`` (e.g. processing chain files,
hit config files).

Adding a New DSP Parameter Script
----------------------------------

Follow these steps to add a new calibration step at the DSP parameter level.

Step 1 – Write the Python function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new file under
``src/legenddataflowscripts/par/geds/dsp/my_step.py``
(or an equivalent path for a new detector type):

.. code-block:: python

   from __future__ import annotations

   import argparse
   from pathlib import Path

   from dbetto.catalog import Props

   from ....utils import build_log


   def par_geds_dsp_my_step() -> None:
       """One-line description of what this step does.

       Extended description: algorithm, inputs, outputs, etc.

       Notes
       -----
       **Command-line arguments**

       ``--config-file`` : list of str
           Configuration file(s) for this step.
       ``--output-file`` : str
           Output parameter file (JSON/YAML).
       ``--log`` : str, optional
           Path to the log file.
       ``--log-config`` : str, optional
           Logging configuration file.
       """
       argparser = argparse.ArgumentParser(description="My new DSP calibration step")
       argparser.add_argument("--config-file", nargs="*", required=True)
       argparser.add_argument("--output-file", required=True)
       argparser.add_argument("--log", default=None)
       argparser.add_argument("--log-config", default={})
       args = argparser.parse_args()

       log = build_log(args.log_config, args.log)

       config = Props.read_from(args.config_file)

       if config.get("run_my_step", True):
           # ... calibration logic here ...
           out_dict = {}
           log.info("my_step complete")
       else:
           out_dict = {}

       Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
       Props.write_to(args.output_file, out_dict)

Key conventions:

* The function must accept **no arguments** (all inputs come from
  ``argparse``).
* Always call :func:`~legenddataflowscripts.utils.build_log` early so that
  log output is captured from the start.
* Guard the main logic with a ``run_*`` flag in the configuration so that
  Snakemake can create a placeholder output without running the step.
* Create parent directories before writing output files.
* Write docstrings in **NumPy style** (the Sphinx Napoleon extension is
  configured for this).

Step 2 – Register the entry point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add the new callable to ``pyproject.toml`` under
``[project.scripts]``:

.. code-block:: toml

   [project.scripts]
   par-geds-dsp-my-step = "legenddataflowscripts.par.geds.dsp.my_step:par_geds_dsp_my_step"

After editing ``pyproject.toml`` reinstall the package in editable mode so the
new command becomes available:

.. code-block:: console

   $ pip install --editable .

Step 3 – Write the Snakemake rule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add a rule to the appropriate rules file (e.g.
``rules/par_geds_dsp.smk``).  A minimal example:

.. code-block:: text

   rule par_geds_dsp_my_step:
       input:
           raw_files = ...,
           config    = ancient(get_my_step_config(...)),
           database  = ...,
       output:
           par_file  = get_par_path("my_step", ...),
       log: get_log_path("par_geds_dsp_my_step", ...)
       shell:
           execenv_pyexe(config, "par-geds-dsp-my-step") + """
               --config-file {input.config}
               --output-file {output.par_file}
               --log        {log}
           """

Step 4 – Add configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a default configuration file for the new step and add it to the
``TextDB`` structure so that ``db.on(timestamp, system=datatype)`` returns it.
The minimum required key is the ``run_*`` guard flag:

.. code-block:: yaml

   # configs/my_step/my_step_config.yaml
   run_my_step: true
   # ... other parameters ...

Step 5 – Write tests
~~~~~~~~~~~~~~~~~~~~~

Add unit or integration tests under ``tests/``.  At minimum, verify that:

* The script exits cleanly when ``run_my_step: false``.
* The output file is created and has the expected structure.

Adding a New HIT Calibration Script
-------------------------------------

The procedure is identical to a new DSP parameter script, with the following
differences:

* Place the file under ``src/legenddataflowscripts/par/geds/hit/``.
* Use the naming convention ``par_geds_hit_my_step`` / ``par-geds-hit-my-step``.
* The script will typically consume the DSP tier LH5 files and the output of
  preceding HIT calibration steps (energy calibration objects, etc.).
* Calibration objects that should be pickled and reused in later steps (e.g.
  for A/E or LQ) must be saved with ``pickle.dump``.

Typical input/output pattern for a HIT script:

.. code-block:: python

   # Input  – DSP LH5 files (via file list) + previous HIT pars
   # Output – updated hit_pars.yaml  +  calibration object .pkl

   from pygama.pargen.utils import load_data

   data, mask = load_data(
       files,
       table_name,
       cal_dict,
       params=params,
       threshold=threshold,
       return_selection_mask=True,
   )
   # ... run calibration ...
   Props.write_to(args.hit_pars, output_dict)
   with Path(args.results).open("wb") as f:
       pickle.dump(calibration_object, f)

Adding a New Detector Type
---------------------------

To support a new detector type (e.g. ``sipms``), create a parallel directory
structure under ``src/legenddataflowscripts/par/sipms/`` following the same
pattern as ``par/geds/``.  Register the new entry points in ``pyproject.toml``
using the naming convention ``par-sipms-{tier}-{step}``.

Code Style and Conventions
---------------------------

* Formatting is enforced by `Black <https://black.readthedocs.io>`_.
* Linting is performed by `Ruff <https://docs.astral.sh/ruff/>`_.
* Imports are sorted with `isort <https://pycqa.github.io/isort/>`_
  (enforced by ``pre-commit``).
* Docstrings **must** use the **NumPy style**; the Sphinx Napoleon extension
  is configured to reject Google-style docstrings.
* Type annotations on function signatures are encouraged but not mandatory.

Run the full pre-commit suite before opening a pull request:

.. code-block:: console

   $ pip install pre-commit
   $ pre-commit run --all-files
