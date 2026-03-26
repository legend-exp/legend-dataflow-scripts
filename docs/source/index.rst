legend-dataflow-scripts
=======================

*legend-dataflow-scripts* is a Python package based on
`Snakemake <https://snakemake.readthedocs.io/en/stable/index.html>`_ for
running the LEGEND data production pipeline.  It calibrates and optimises
hundreds of HPGe detector channels in parallel before building the final
analysis-ready data tiers.  It takes as input the detector metadata from
`legend-metadata <https://github.com/legend-exp/legend-metadata>`_.

Getting started
---------------

It is recommended to install the package using uv.

For a quick development install::

   uv pip install ".[docs,test]"

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   Package overview <overview>
   CLI entry points <callables>

:doc:`overview`
   Architecture, module descriptions, and dependency overview.

:doc:`callables`
   All command-line tools provided by the package with descriptions and usage
   examples.

API Reference
-------------

.. toctree::
   :maxdepth: 1
   :caption: API reference

   Package API reference <api/modules>

Development
-----------

.. toctree::
   :maxdepth: 2
   :caption: Development

   Developer guide <developer>
   Source Code <https://github.com/legend-exp/legend-dataflow>

:doc:`developer`
   How to add new calibration scripts, Snakemake rules, and entry points.

Related Projects
----------------

.. toctree::
   :maxdepth: 1
   :caption: Related projects

   LEGEND Data Objects <https://legend-pydataobj.readthedocs.io>
   Decoding Digitizer Data <https://legend-daq2lh5.readthedocs.io>
   Digital Signal Processing <https://dspeed.readthedocs.io>
   Pygama <https://pygama.readthedocs.io>
