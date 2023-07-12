.. _sdp-introduction:

Speech Data Processor
=====================

Speech Data Processor (SDP) is a toolkit to make it easy to:

1. Write code to process a new dataset, minimizing the amount of boilerplate code required.
2. Share the steps for processing a speech dataset.

SDP is hosted here: https://github.com/NVIDIA/NeMo-speech-data-processor.
It's mainly used to prepare datasets for `NeMo toolkit <https://github.com/NVIDIA/NeMo>`_.

SDP's philosophy is to represent processing operations as 'processor' classes, which take in a path to a NeMo-style
data manifest as input (or a path to the raw data directory if you do not have a NeMo-style manifest to start with),
apply some processing to it, and then save the output manifest file.

You specify which processors you want to run using a YAML config file. Many common processing operations are provided,
and it is easy to add your own.

.. image:: https://github.com/NVIDIA/NeMo/releases/download/v1.17.0/sdp_overview_diagram.png
   :alt: Overview diagram of Speech Data Processor

To learn more about SDP, have a look at the following sections.

.. toctree::
   :maxdepth: 1

   sdp/config_structure
   sdp/adding_processors
   sdp/existing_configs
   sdp/api
