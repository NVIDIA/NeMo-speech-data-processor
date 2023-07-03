.. _sdp-introduction:

Speech Data Processor
========================

Speech Data Processor (SDP) is a toolkit to make it easy to:
  1. write code to process a new dataset, minimizing the amount of boilerplate code required.
  2. share the steps for processing a speech dataset.

SDP is hosted here: https://github.com/NVIDIA/NeMo-speech-data-processor.

SDP's philosophy is to represent processing operations as 'processor' classes, which take in a path to a NeMo-style
data manifest as input (or a path to the raw data directory if you do not have a NeMo-style manifest to start with),
apply some processing to it, and then save the output manifest file.

You specifiy which processors you want to run using a YAML config file. Many common processing operations are provided,
and it is easy to add your own. If you do not need to add your own processors, then all that is needed to process a
new dataset is to write a single YAML file containing the parameters needed to process your dataset.

.. image:: https://github.com/NVIDIA/NeMo/releases/download/v1.17.0/sdp_overview_diagram.png
   :alt: Overview diagram of Speech Data Processor

After you created a config file (or re-used `one of the existing
ones <https://github.com/NVIDIA/NeMo-speech-data-processor/tree/main/dataset_configs>`_) you only need to run::

  python main.py \
    --config-path <path to the config folder> \
    --config-name <config file name> \
    <any other supported arguments>

You can run the script with ``--help`` argument to see all available config parameters.

To learn more about SDP, have a look at the following sections.

.. toctree::
   :maxdepth: 1

   config_structure
   adding_processors
   existing_configs
   architecture_description
   api
