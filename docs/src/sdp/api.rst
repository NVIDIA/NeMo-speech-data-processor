API
---

Available processors
~~~~~~~~~~~~~~~~~~~~

Here is the full list of all available processors and their supported arguments.

.. Using autodata everywhere to only have class name and docs and save on space


Dataset-specific processors
###########################

MCV
'''

.. autodata:: sdp.processors.CreateInitialManifestMCV
   :annotation:

MLS
'''

.. autodata:: sdp.processors.CreateInitialManifestMLS
   :annotation:

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

.. TODO: Should look through all the classes to make sure docs list all parameters with enough details


Data enrichment
###############

The following processors can be used to add additional attributes to the data by
running different NeMo models (e.g., ASR predictions). These attributes are typically
used in the downstream processing for additional enhancement or filtering.

.. autodata:: sdp.processors.ASRInference
   :annotation:

.. autodata:: sdp.processors.PCInference
   :annotation:


Text-only modifications
#######################

Data modifications
''''''''''''''''''

.. autodata:: sdp.processors.SubRegex
   :annotation:

.. autodata:: sdp.processors.SubMakeLowercase
   :annotation:

.. autodata:: sdp.processors.MakeLettersUppercaseAfterPeriod
   :annotation:

.. autodata:: sdp.processors.ChangePCFields
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


ASR-based modifications
#######################

.. note::
    All processors in this section depend on the :class:`sdp.processors.ASRInference`.
    So make sure to include it in the config at some prior stage with an applicable
    ASR model.

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

.. autodata:: sdp.processors.DropHighCER
   :annotation:

.. autodata:: sdp.processors.DropHighLowCharrate
   :annotation:

.. autodata:: sdp.processors.DropHighLowDuration
   :annotation:

.. autodata:: sdp.processors.DropHighLowWordrate
   :annotation:

.. autodata:: sdp.processors.DropIfSubstringInInsertion
   :annotation:

.. autodata:: sdp.processors.DropLowWordMatchRate
   :annotation:


Miscellaneous
#############

.. autodata:: sdp.processors.ChangePCFields
   :annotation:

.. autodata:: sdp.processors.AddConstantFields
   :annotation:

.. autodata:: sdp.processors.ChangeToRelativePath
   :annotation:

.. autodata:: sdp.processors.DuplicateFields
   :annotation:

.. autodata:: sdp.processors.RenameFields
   :annotation:

.. autodata:: sdp.processors.SplitOnFixedDuration
   :annotation:

.. autodata:: sdp.processors.WriteManifest
   :annotation:


Base classes
~~~~~~~~~~~~

This section lists all the base classes you might need to know about if you
want to add new SDP processors.

BaseProcessor
#############

.. autoclass:: sdp.processors.base_processor.BaseProcessor
   :show-inheritance:

BaseParallelProcessor
#####################

.. autoclass:: sdp.processors.base_processor.BaseParallelProcessor
   :show-inheritance:

ModifyManifestTextProcessor
###########################

.. autoclass:: sdp.processors.base_processor.BaseParallelProcessor
   :show-inheritance:
