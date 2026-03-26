CLI Entry Points (Callables)
============================

All CLI commands are registered as `entry points
<https://packaging.python.org/en/latest/specifications/entry-points/>`_ in
``pyproject.toml`` and become available as shell commands after the package is
installed.  Each command wraps a single Python function; click on the function
name to see its full documentation.

Workflow Management
-------------------

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Command
     - Python function
     - Description
   * - ``dataflow``
     - :func:`~legenddataflowscripts.workflow.execenv.dataflow`
     - Top-level CLI for installing software and executing commands inside the
       data-production environment (supports ``install`` and ``exec``
       sub-commands).
   * - ``build-filedb``
     - :func:`~legenddataflowscripts.workflow.filedb.build_filedb`
     - Scan a directory of raw LH5 files and build a
       :class:`~pygama.flow.file_db.FileDB` with per-file first timestamps.

Data Tier Building
------------------

.. list-table::
   :header-rows: 1
   :widths: 45 30 25

   * - Command
     - Python function
     - Description
   * - ``build-tier-dsp``
     - :func:`~legenddataflowscripts.tier.dsp.build_tier_dsp`
     - Build the DSP tier for all channels in a run file; supports
       multiprocessing.
   * - ``build-tier-dsp-single-channel``
     - :func:`~legenddataflowscripts.tier.dsp.build_tier_dsp_single_channel`
     - Build the DSP tier for a single detector channel.
   * - ``build-tier-hit``
     - :func:`~legenddataflowscripts.tier.hit.build_tier_hit`
     - Build the HIT tier for all channels from DSP output.
   * - ``build-tier-hit-single-channel``
     - :func:`~legenddataflowscripts.tier.hit.build_tier_hit_single_channel`
     - Build the HIT tier for a single detector channel.

DSP Parameter Optimisation (``par-geds-dsp-*``)
------------------------------------------------

These scripts are run during the calibration phase to determine optimal digital
signal processing parameters for each HPGe detector channel.  They are
typically executed in the order shown below.

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Command
     - Python function
     - Description
   * - ``par-geds-dsp-pz``
     - :func:`~legenddataflowscripts.par.geds.dsp.pz.par_geds_dsp_pz`
     - Fit the pole-zero decay constant(s) from waveform tail slopes.
       Supports single-exponential (``mode: single``) and
       double-exponential (``mode: double``) models.
   * - ``par-geds-dsp-nopt``
     - :func:`~legenddataflowscripts.par.geds.dsp.nopt.par_geds_dsp_nopt`
     - Optimise DSP filter shaping for noise rejection using low-energy
       baseline events.
   * - ``par-geds-dsp-evtsel``
     - :func:`~legenddataflowscripts.par.geds.dsp.evtsel.par_geds_dsp_evtsel`
     - Select calibration peak events from raw data files for use by the
       energy optimiser and DPLMS filter builder.
   * - ``par-geds-dsp-eopt``
     - :func:`~legenddataflowscripts.par.geds.dsp.eopt.par_geds_dsp_eopt`
     - Bayesian optimisation of CUSP, ZAC, and trapezoidal filter shaping
       parameters for best energy resolution.
   * - ``par-geds-dsp-dplms``
     - :func:`~legenddataflowscripts.par.geds.dsp.dplms.par_geds_dsp_dplms`
     - Compute DPLMS optimal filter coefficients from FFT baseline and
       calibration peak data.
   * - ``par-geds-dsp-svm-build``
     - :func:`~legenddataflowscripts.par.geds.dsp.svm_build.par_geds_dsp_svm_build`
     - Train a Support Vector Machine classifier on discrete wavelet transform
       features for pulse-shape discrimination.
   * - ``par-geds-dsp-svm``
     - :func:`~legenddataflowscripts.par.geds.dsp.svm.par_geds_dsp_svm`
     - Register a pre-trained SVM model file path in the DSP parameter
       database.

HIT Parameter Optimisation (``par-geds-hit-*``)
------------------------------------------------

These scripts are run after the DSP tier is built to calibrate physics
observables at the hit level.

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Command
     - Python function
     - Description
   * - ``par-geds-hit-qc``
     - :func:`~legenddataflowscripts.par.geds.hit.qc.par_geds_hit_qc`
     - Derive data-driven quality-cut classifiers from calibration and FFT
       run data.
   * - ``par-geds-hit-ecal``
     - :func:`~legenddataflowscripts.par.geds.hit.ecal.par_geds_hit_ecal`
     - HPGe energy calibration: peak finding, fitting, and FWHM curve
       extraction for multiple energy parameters.
   * - ``par-geds-hit-aoe``
     - :func:`~legenddataflowscripts.par.geds.hit.aoe.par_geds_hit_aoe`
     - Calibrate the A/E (current amplitude over energy) pulse-shape
       discriminant with optional drift-time and energy-dependent corrections.
   * - ``par-geds-hit-lq``
     - :func:`~legenddataflowscripts.par.geds.hit.lq.par_geds_hit_lq`
     - Calibrate the LQ (late charge) pulse-shape discriminant with DEP-based
       cut determination.

Typical Invocation
------------------

Commands are usually invoked by Snakemake rules, but can be called directly
for debugging:

.. code-block:: console

   $ par-geds-dsp-pz \
       --processing-chain dsp_config.yaml \
       --config-file pz_config.yaml \
       --raw-table-name ch1057600/raw \
       --raw-files /data/raw/run001.lh5 \
       --output-file /output/pz_pars.yaml

   $ build-tier-dsp \
       --configs /path/to/configs \
       --datatype cal \
       --timestamp 20230401T000000Z \
       --tier dsp \
       --input /data/raw/run001.lh5 \
       --output /data/dsp/run001.lh5 \
       --n-processes 4
