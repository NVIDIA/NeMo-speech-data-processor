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
ones <https://github.com/NVIDIA/NeMo-speech-data-processor/tree/main/dataset_configs>`_) you only need to run

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

.. Processor classes
.. -----------------

.. **BaseProcessor**
.. ~~~~~~~~~~~~~~~~~

.. All processor classes inherit from the ``BaseProcessor`` class. This is a simple abstract class which has 2 empty methods: ``process()`` and ``test()``.
.. These serve to remind us that SDP essentially just runs ``test()`` on all processors, and then ``process()`` on all processors (more details about testing :ref:`here<SDP Tests>`).

.. ``ASRInference`` is a child class of ``BaseProcessor``. It has a simple ``process()`` method which runs transcription on every utterance in the input_manifest.

.. ``WriteManifest`` is also a child class of ``BaseProcessor``. It has a simple ``process()`` method which saves a copy of the input manifest containing only the fields specified in ``fields_to_save``.

.. **BaseParallelProcessor**
.. ~~~~~~~~~~~~~~~~~~~~~~~~~
.. ``BaseParallelProcessor`` inherits from the ``BaseProcessor`` class. Within the ``BaseParallelProcessor.process()`` method, it calls other methods and functions, which allow it to do more complex processing.
.. Most importantly, it calls its ``BaseParallelProcessor.process_dataset_entry(data_entry)`` method on every utterance in the manifest, and it does this in parallel, allowing for more efficient processing.

.. What is a **DataEntry**?
.. ~~~~~~~~~~~~~~~~~~~~~~~~
.. As mentioned above, ``BaseParallelProcessor.process_dataset_entry(data_entry)`` is called on a variable called ``data_entry`` which represents an utterance in our dataset.
.. Most often, ``data_entry`` will be a dictionary containing items which represent the JSON manifest entry.
.. Sometimes, such as in ``CreateInitialManifestMLS``, it will be a string containing a line for that utterance from the original raw MLS transcript.

.. ``BaseParallelProcessor.process_dataset_entry`` will process ``data_entry`` and output a ``DataEntry`` object.

.. The ``DataEntry`` class is a dataclass which contains 2 attributes:

.. 1. ``data`` is an Optional dictionary containing items which represent the JSON manifest entry. ``data`` can also be ``None``. If a ``.process_dataset_entry(data_entry)`` method returns a ``DataEntry`` class where ``data is None``, then that utterance will be dropped from the output manifest.
.. 2. ``metrics``, which can be of any type, and are ``None`` by default. This variable is used by some variables to record summary statistics about the changes made to the dataset, these metrics are aggregated and can be displayed once every utterance has been processed by the processor.

.. What happens in **BaseParallelProcessor.process()**?
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. We outline the ``BaseParallelProcessor.process()`` method below:

.. .. raw:: html

..     <div align="center">
..       <img src="https://mermaid.ink/img/pako:eNplUl1r6zAM_SvCFy4pbL3vvaVwu-59sL0tl6LESmqIP7DkjWzsv89O0rVjzosiHR8dHetdtV6T2qg-YjjB0-Fv7SAfTs2cqdWjUGAwDrYiuz0yPWDEYaDhIfqWmH1chzmqVts_GQOW5OR1rWaqcv4916pcZxq6jKaAkRb0tok7IBtkXO5BM4KmDtMgUIotOmgIEpMG8VOK1v0atH91g0cNEV9BoyBgEm9RTJvljbX6D7e3O9hfVOyvVURCfbToTEcs11pKocwbksC5PnWFyhB00VvIE7wYnxiWwY3rgbNNqwlnOpATRQLD4B2dhdxdhNx9t2PiOJYRmORITuJYlb85XEydFGDDErGVL4tn6gNcuA-Zm_GFwCf5McJvwL6P1KNQoYim5SlfTY7-At9BEmHQ0YdAenVucH_hv7_W3hmHg3mj40JWXYudX8lwGHD86rb4d7YtN6hd-Qo1Oa1ulKVo0ei8k-8lXatsps0ubnK47EVZrY8MLQ_-OLpWbSQmulEpZNvoYDDvrlWbDgemj0-10vX9" height=100% />
..     </div>


.. **ModifyManifestTextProcessor**
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. ``ModifyManifestTextProcessor`` inherits from the ``BaseParallelProcessor`` class.

.. The ``ModifyManifestTextProcessor`` constructor takes in the following arguments:
.. * ``text_key`` (string) and ``pred_text_key`` (string): these parameters specify which keys in ``data_entry.data`` will be used for processing. (default: ``text_key="text"``, ``pred_text_key="pred_text"``, ie. by default the processor will refer to and modify the ``"text"`` and/or ``"pred_text"`` attributes of the input manifest).
.. * ``test_cases`` (optional, list of dicts) - test cases for checking that the processor makes the changes that we are expecting.

.. ``ModifyManifestTextProcessor`` has the following methods:
.. * ``ModifyManifestTextProcessor.test()``: this method makes sure that the output from the processor matches the expected output specified in the ``test_cases`` parameter.
.. * ``ModifyManifestTextProcessor.process_dataset_entry(data_entry)``: this method applies processing to a ``data_entry``. First, spaces are added to the start and end of the 'text' and 'pred_text' entries (if they exist), then the abstract method ``ModifyManifestTextProcessor._process_dataset_entry(data_entry)`` is called. Then, any extra spaces (e.g. two spaces next to each other '  ') are removed from 'text' and 'pred_text' entries.
.. * ``ModifyManifestTextProcessor._process_dataset_entry(data_entry)``: this is an abstract method which will be over-written by children of ``ModifyManifestTextProcessor``.
