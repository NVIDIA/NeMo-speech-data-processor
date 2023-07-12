API
---

Available processors
~~~~~~~~~~~~~~~~~~~~

Here is the full list of all available processors and their supported arguments.

.. note::
    All SDP processors optionally accept ``input_manifest_file`` and
    ``output_manifest_file`` keys. See :ref:`Special fields <special_fields>` section
    for more details.

.. Using autodata everywhere to only have class name and docs and save on space


Dataset-specific processors
###########################

MCV
'''

.. autodata:: sdp.processors.CreateInitialManifestMCV
   :annotation:

MLS
'''

.. indexed in the adding_processors section with methods description
.. autodata:: sdp.processors.CreateInitialManifestMLS
   :annotation:
   :noindex:

.. autodata:: sdp.processors.RestorePCForMLS
   :annotation:

VoxPopuli
'''''''''

.. autodata:: sdp.processors.CreateInitialManifestVoxpopuli
   :annotation:

.. autodata:: sdp.processors.NormalizeFromNonPCTextVoxpopuli
   :annotation:

CORAAL
''''''

.. autodata:: sdp.processors.CreateInitialManifestCORAAL
   :annotation:

.. autodata:: sdp.processors.TrainDevTestSplitCORAAL
   :annotation:

SLR83
'''''

.. autodata:: sdp.processors.CreateInitialManifestSLR83
   :annotation:

.. autodata:: sdp.processors.CustomDataSplitSLR83
   :annotation:

.. TODO: Fisher config is not accessible - should we require moving everything to SDP
..       Probably need some policy on shat lives in main folder vs configs.
..       To control the number of processors we support.

Data enrichment
###############

The following processors can be used to add additional attributes to the data by
running different NeMo models (e.g., ASR predictions). These attributes are typically
used in the downstream processing for additional enhancement or filtering.

.. autodata:: sdp.processors.ASRInference
   :annotation:

.. autodata:: sdp.processors.PCInference
   :annotation:


Text-only processors
####################

.. note::
    All processors in this section accept additional parameter
    ``text_key`` (defaults to "text") to control which field is used
    for modifications/filtering.

Data modifications
''''''''''''''''''

.. indexed in the adding_processors section with methods description

.. autodata:: sdp.processors.SubRegex
   :annotation:
   :noindex:

.. autodata:: sdp.processors.SubMakeLowercase
   :annotation:

.. autodata:: sdp.processors.MakeLettersUppercaseAfterPeriod
   :annotation:

Data filtering
''''''''''''''

.. autodata:: sdp.processors.DropIfRegexMatch
   :annotation:

.. autodata:: sdp.processors.DropIfNoneOfRegexMatch
   :annotation:

.. autodata:: sdp.processors.DropNonAlphabet
   :annotation:

.. autodata:: sdp.processors.DropOnAttribute
   :annotation:


ASR-based processors
####################

.. note::
    All processors in this section depend on the :class:`sdp.processors.ASRInference`.
    So make sure to include it in the config at some prior stage with an applicable
    ASR model.

.. note::
    All processors in this section accept additional parameters
    ``text_key`` (defaults to "text") and ``pred_text_key`` (defaults to "text_pred")
    to control which fields contain transcription and ASR model predictions.

Data modifications
''''''''''''''''''

.. autodata:: sdp.processors.InsIfASRInsertion
   :annotation:

.. autodata:: sdp.processors.SubIfASRSubstitution
   :annotation:

Data filtering
''''''''''''''

.. autodata:: sdp.processors.DropASRError
   :annotation:

.. autodata:: sdp.processors.DropASRErrorBeginningEnd
   :annotation:

.. autodata:: sdp.processors.DropIfSubstringInInsertion
   :annotation:

.. autodata:: sdp.processors.DropHighCER
   :annotation:

.. autodata:: sdp.processors.DropHighWER
   :annotation:

.. autodata:: sdp.processors.DropLowWordMatchRate
   :annotation:

.. indexed in the adding_processors section with methods description
.. autodata:: sdp.processors.DropHighLowCharrate
   :annotation:
   :noindex:

.. autodata:: sdp.processors.DropHighLowWordrate
   :annotation:

.. autodata:: sdp.processors.DropHighLowDuration
   :annotation:


Miscellaneous
#############

.. autodata:: sdp.processors.AddConstantFields
   :annotation:

.. autodata:: sdp.processors.CombineSources
   :annotation:

.. autodata:: sdp.processors.DuplicateFields
   :annotation:

.. autodata:: sdp.processors.RenameFields
   :annotation:

.. autodata:: sdp.processors.SplitOnFixedDuration
   :annotation:

.. autodata:: sdp.processors.ChangeToRelativePath
   :annotation:

.. autodata:: sdp.processors.SortManifest
   :annotation:

.. autodata:: sdp.processors.WriteManifest
   :annotation:


.. _sdp-base-classes:

Base classes
~~~~~~~~~~~~

This section lists all the base classes you might need to know about if you
want to add new SDP processors.

BaseProcessor
#############

.. autoclass:: sdp.processors.base_processor.BaseProcessor
   :show-inheritance:
   :private-members:
   :member-order: bysource
   :exclude-members: _abc_impl

BaseParallelProcessor
#####################

.. autoclass:: sdp.processors.base_processor.BaseParallelProcessor
   :show-inheritance:
   :private-members:
   :member-order: bysource
   :exclude-members: _abc_impl

ModifyManifestTextProcessor
###########################

.. autoclass:: sdp.processors.modify_manifest.modify_manifest.ModifyManifestTextProcessor
   :show-inheritance:
   :private-members:
   :member-order: bysource
   :no-inherited-members:
   :exclude-members: _abc_impl

.. _sdp-runtime-tests:

Runtime tests
#############

Before running the specified processors, SDP runs ``processor.test()`` on all specified processors.
Currently, the only provided processor classes with a test method are subclasses of
:class:`sdp.processors.modify_manifest.modify_manifest.ModifyManifestTextProcessor`.

:meth:`sdp.processors.modify_manifest.modify_manifest.ModifyManifestTextProcessor.test`
runs any ``test_cases`` that were provided in the object constructor.
This means you can provided test cases in the YAML config file, and the
dataset will only be processed if the test cases pass.

This is helpful to (a) make sure that the rules you wrote have the effect
you desired, and (b) demonstrate why you wrote those rules.
An example of test cases we could include in the YAML config file::

    - _target_: sdp.processors.DropIfRegexMatch
    regex_patterns:
        - "(\\D ){5,20}" # looks for between 4 and 19 characters surrounded by spaces
    test_cases:
        - {input: {text: "some s p a c e d out letters"}, output: null}
        - {input: {text: "normal words only"}, output: {text: "normal words only"}}
