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


Text-only modifications
#######################


Data filtering
##############


Miscellaneous
#############


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
