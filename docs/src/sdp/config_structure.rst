How to write config files?
--------------------------

The YAML config file for processing a dataset must contain a key ``processors``, the value of which is a list.
Each item in that list is expected to be a dictionary specifying a processor class, i.e. it must have a key
``_target_``, the value of which is a path to a "processor" class, and the remaining keys must be the kwargs
necessary to instantiate that class with `hydra.utils.instantiate() <https://hydra.cc/docs/advanced/instantiate_objects/overview/>`_.

SDP will run the processors specified in the ``processors`` list in the config file. It will also check for a
``processors_to_run`` key in the config file, which can be either the string ``all``, or any Python "slice" object
like ``3:4``, ``2:`` etc. (if there is no ``processors_to_run`` key, then all of the processors will be run).

.. note::
    SDP will run the processors in the order in which they are listed in the config YAML file. Make sure to list the
    processors in an order which makes sense, e.g. create an initial manifest first; make sure to run ASR inference
    before doing any processing which looks at ``pred_text`` fields in the manifest.

For an example of the config file, see the :ref:`introduction <sdp-introduction>` or have a look at one of the many
config files in https://github.com/NVIDIA/NeMo-speech-data-processor/tree/main/dataset_configs.

.. _special_fields:

Special fields
~~~~~~~~~~~~~~

There are a few special fields that SDP allows to add or modifies, besides all the other arguments of
the processors you're using.

* **input_manifest_file/output_manifest_file (str)**: virtually all SDP processors accept ``input_manifest_file`` and
  ``output_manifest_file`` arguments to specify where the input is coming from and where to save the output.
  Since most often the input to the next processor is the output of the current processor, you can skip specifying
  those arguments for any processors and SDP will automatically "stitch" consecutive processors together by creating
  temporary files. So most often you only need to specify those arguments for the final processors or any other processors
  that you need to retain an output for (either for caching of the costly computation or to inspect the output for
  debugging purposes). If specified, the ``input_manifest_file`` and ``output_manifest_file`` for a particular processor
  cannot be the same.

.. note::
  SDP fills in any unspecified ``input_manifest_file`` and ``output_manifest_file`` arguments by looping over all
  processors that :ref:`should run <should_run>`, and, for each processor, creating a temporary file for the
  ``output_manifest_file`` if it is unspecified, and linking it to the next processor as the ``input_manifest_file``
  if that is unspecified.

.. _should_run:

* **should_run (bool)**: this boolean field allows to skip any processors in the config. It can be useful to either
  temporarily skip the optional processors or to add certain conditions on when the processors should run, using the
  :ref:`custom resolvers <custom_resolvers>`.

* **test_cases (list[dict])**: most of the processors support a special ``test_cases`` argument.
  It does not change the processor behavior in any way, but is a useful feature to make sure
  the processors are going to work as you expect. The format of this argument is to provide a list
  of dictionaries indicating the input data and corresponding output data. E.g.::

      - _target_: sdp.processors.SubRegex
        regex_params_list:
          - {"pattern": "!", "repl": "."}
          - {"pattern": ";", "repl": ""}
          - {"pattern": " www\\.(\\S)", "repl" : ' www punto \1'}
          - {"pattern": "(\\S)\\.com ", "repl" : '\1 punto com '}
        test_cases:
          - {input: {text: "www.abc.com"}, output: {text: "www punto abc punto com"}}
          - {input: {text: "hey!"}, output: {text: "hey."}}
          - {input: {text: "hey;"}, output: {text: "hey."}}

  or another example::

      - _target_: sdp.processors.DropIfRegexMatch
        regex_patterns:
          - "(\\D ){5,20}" # looks for between 4 and 19 characters surrounded by spaces
        test_cases:
          - {input: {text: "some s p a c e d out letters"}, output: null}
          - {input: {text: "normal words only"}, output: {text: "normal words only"}}

  Regular expressions can be tricky to get right and so you can provide any number
  of examples that we will run through the processor to make sure that all inputs
  map to the desired outputs.

.. _custom_resolvers:

Custom resolvers
~~~~~~~~~~~~~~~~

We define a few custom `OmegaConf resolvers <https://omegaconf.readthedocs.io/en/latest/usage.html#resolvers>`_
to simplify common operations useful across many config files.

* **subfield**: can be used to add conditions to the config files. It uses the following syntax::

    ${subfield:<dictionary>,<field to select>}

  E.g., you can use this resolver to pick the right parameter value for each data split::

    # data_split should be provided by the user in a command-line argument
    data_split: ???

    # first we define a dictionary that holds the parameters
    high_duration_thresholds:
      train: 20
      dev: 25
      test: 30

    processors:
      ...

      # then we use the subfield resolver to pick the right
      # value from this dictionary as an argument
      - _target_: sdp.processors.DropHighLowDuration
        high_duration_threshold: ${subfield:${high_duration_thresholds},${data_split}}
        ...
* **not**: can be used to negate the boolean arguments of the config file. It uses the following syntax::

    ${not:<parameter to negate>}

  E.g., if you have a parameter that's used to select when a certain
  processor should run, but some other processors require a negation of
  that parameter, you can use a ``not`` resolver to simplify the logic::

    # can be used to control if we need to restore punctuation and capitalization
    restore_pc: True

    processors:
      ...

      # for one processor we want ot use the value directly
      - _target_: sdp.processors.NormalizeFromNonPCTextVoxpopuli
        should_run: ${restore_pc}

      ...

      # but for another we might need to specify a negation of the argument
      - _target_: sdp.processors.SubMakeLowercase
        should_run: ${not:${restore_pc}}

* **equal**: can be used to compare argument to another argument or constant. It uses the following syntax::

    ${equal:<argument to compare>,<value for comparison>}

  E.g., you can use this resolver to create more complex config flows by allowing
  multiple values to control which processors should run.
  See `Italian MLS (with P&C) config file <https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/dataset_configs/italian/mls/config.yaml>`_
  for an example.


Tips for writing effective configs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Skip "input_manifest_file" and "output_manifest_file" unless you need them.**

For most of the configs you can completely skip the input manifests unless you need to support
non-linear processor flow (e.g., for saving parts of the manifest file to different data splits as done in the
`CORAAL config file <https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/dataset_configs/english/coraal/config.yaml>`_).

You always need to explicitly specify output manifest for the final processor. The other good use-case for manually
specifying it is to "cache" outputs of the expensive processors. This can be done if you expect that you'd need
to iterate on running config file multiple times tweaking different parameters of the processors. If that's the case,
make sure to save the output of the expensive processors, so that you can restart from those processors without
rerunning them. For example, `Italian MCV config file <https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/dataset_configs/italian/mcv/config.yaml>`_
caches the output of the first processor, so that you can later re-run it with added ``processors_to_run="1:"`` and
the costly initial manifest creation can be fully re-used.

**Add conditions to the configs.**

There are two common examples of the conditions we might want to support.

* We can have different parameters for the processors based on the data split. E.g., in the
  `Spanish VoxPopuli config file <https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/dataset_configs/spanish_pc/voxpopuli/config.yaml>`_
  we have different thresholds specified in the ``high_duration_thresholds`` dictionary that are later used
  in the ``DropHighLowDuration`` processor.
* We can skip some of the processors based on the data split specified by the user. E.g., in the
  `Italian MLS config file <https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/dataset_configs/italian/mls/config.yaml>`_
  we skip all the filtering processors for both validation and test splits to ensure we don't modify the provided
  dev/test data to enable fair comparison with prior works.

**Write run-time tests.**

Most SDP processors support run-time tests with a ``test_cases`` argument. Make sure to utilize it
when you create new configs. It can be very helpful to ensure that what you have in the config does
indeed work as you intended. All of our configs have test cases included, so any file is good to
look at as an example.

For more information about the run-time tests see :ref:`run-time tests <sdp-runtime-tests>`.

**Use "local" processors.**

If you need to add some additional functionality to SDP, you don't need to modify the source code.
Instead, you can just create a separate file anywhere you want and then use the full path
to that file in the ``_target_`` section of the config file. E.g., have a look at
`Spanish MLS config file <https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/dataset_configs/spanish/mls/config.yaml>`_
for an example.

.. note::
  You might need to change python path if you get import errors with "local" processors.
  You can do that by prepending::

    PYTHONPATH=<path to the code folder> python main.py ...