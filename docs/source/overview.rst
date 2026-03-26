Package Overview
================

*legend-dataflow-scripts* provides the Python scripts and library utilities
that power the LEGEND-200 data production pipeline.  The package is designed
to calibrate and optimise hundreds of HPGe detector channels in parallel and
then merge the results before building the final analysis-ready data tiers.

.. note::

   The package is intended to be run through the `legend-prodenv
   <https://github.com/legend-exp/legend-prodenv>`_ production environment,
   which manages software installation and container execution.  Direct
   invocation of the CLI entry points is possible for development and testing.

Architecture
------------

Data are processed in sequential *tiers*, each represented by an LH5
(HDF5-based) file.  The pipeline is:

.. code-block:: text

   Raw detector data (raw tier)
           │
           ▼
   ┌───────────────────────────┐
   │  DSP parameter optimisation│
   │  (par/geds/dsp/)           │
   │  - PZ correction           │
   │  - Noise optimisation      │
   │  - Energy optimisation     │
   │  - Event selection         │
   │  - DPLMS filter            │
   │  - SVM classifier          │
   └──────────┬────────────────┘
              │  par_dsp.yaml
              ▼
   ┌───────────────────────────┐
   │    build-tier-dsp          │  (tier/dsp.py)
   │    dspeed processing       │
   └──────────┬────────────────┘
              │  dsp LH5 file
              ▼
   ┌───────────────────────────┐
   │  HIT parameter optimisation│
   │  (par/geds/hit/)           │
   │  - Quality cuts            │
   │  - Energy calibration      │
   │  - A/E calibration         │
   │  - LQ calibration          │
   └──────────┬────────────────┘
              │  par_hit.yaml
              ▼
   ┌───────────────────────────┐
   │    build-tier-hit          │  (tier/hit.py)
   │    pygama hit builder      │
   └──────────┬────────────────┘
              │  hit LH5 file
              ▼
         Physics analysis

Module Reference
----------------

The package is organised into four sub-packages:

.. contents::
   :local:
   :depth: 1

``legenddataflowscripts.tier``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data-tier building scripts.  Each function is a self-contained CLI entry point
that reads one tier of LH5 data and writes the next.

.. autosummary::
   :nosignatures:

   legenddataflowscripts.tier.dsp.build_tier_dsp
   legenddataflowscripts.tier.dsp.build_tier_dsp_single_channel
   legenddataflowscripts.tier.hit.build_tier_hit
   legenddataflowscripts.tier.hit.build_tier_hit_single_channel

``legenddataflowscripts.par``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calibration and parameter-optimisation scripts, organised by detector type
(currently only ``geds`` for germanium detectors) and tier.

**DSP parameter optimisation** (``par.geds.dsp``):

.. autosummary::
   :nosignatures:

   legenddataflowscripts.par.geds.dsp.pz.par_geds_dsp_pz
   legenddataflowscripts.par.geds.dsp.nopt.par_geds_dsp_nopt
   legenddataflowscripts.par.geds.dsp.evtsel.par_geds_dsp_evtsel
   legenddataflowscripts.par.geds.dsp.eopt.par_geds_dsp_eopt
   legenddataflowscripts.par.geds.dsp.dplms.par_geds_dsp_dplms
   legenddataflowscripts.par.geds.dsp.svm_build.par_geds_dsp_svm_build
   legenddataflowscripts.par.geds.dsp.svm.par_geds_dsp_svm

**HIT parameter optimisation** (``par.geds.hit``):

.. autosummary::
   :nosignatures:

   legenddataflowscripts.par.geds.hit.qc.par_geds_hit_qc
   legenddataflowscripts.par.geds.hit.ecal.par_geds_hit_ecal
   legenddataflowscripts.par.geds.hit.aoe.par_geds_hit_aoe
   legenddataflowscripts.par.geds.hit.lq.par_geds_hit_lq

``legenddataflowscripts.utils``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shared utility functions used across the calibration and tier-building scripts.

.. autosummary::
   :nosignatures:

   legenddataflowscripts.utils.log.build_log
   legenddataflowscripts.utils.log.StreamToLogger
   legenddataflowscripts.utils.alias_table.alias_table
   legenddataflowscripts.utils.alias_table.convert_parents_to_structs
   legenddataflowscripts.utils.pulser_removal.get_pulser_mask
   legenddataflowscripts.utils.plot_dict.fill_plot_dict
   legenddataflowscripts.utils.cfgtools.get_channel_config
   legenddataflowscripts.utils.convert_np.convert_dict_np_to_float

``legenddataflowscripts.workflow``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Workflow infrastructure: execution environment management, file-database
construction, configuration variable substitution, and catalog pre-compilation.

.. autosummary::
   :nosignatures:

   legenddataflowscripts.workflow.execenv.execenv_prefix
   legenddataflowscripts.workflow.execenv.execenv_pyexe
   legenddataflowscripts.workflow.filedb.build_filedb
   legenddataflowscripts.workflow.pre_compile_catalog.pre_compile_catalog
   legenddataflowscripts.workflow.utils.subst_vars
   legenddataflowscripts.workflow.utils.subst_vars_in_snakemake_config
   legenddataflowscripts.workflow.utils.set_last_rule_name
   legenddataflowscripts.workflow.utils.as_ro

Key Dependencies
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Package
     - Role
   * - `dspeed <https://dspeed.readthedocs.io>`_
     - Digital signal processing engine used by the DSP tier builder.
   * - `pygama <https://pygama.readthedocs.io>`_
     - Calibration algorithms (energy cal, A/E, LQ, noise optimisation, etc.)
       and the HIT tier builder.
   * - `lgdo / lh5 <https://legend-pydataobj.readthedocs.io>`_
     - LEGEND Data Object format and LH5 file I/O.
   * - `dbetto <https://dbetto.readthedocs.io>`_
     - Configuration file handling (JSON/YAML, TextDB, validity catalogs).
   * - `pylegendmeta <https://pylegendmeta.readthedocs.io>`_
     - LEGEND metadata access.
   * - scikit-learn
     - SVM classifier training and inference.
   * - Snakemake
     - Workflow scheduler that orchestrates all processing rules.
