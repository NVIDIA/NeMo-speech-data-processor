.. Make sure to run docs/gen_docs.py before running any of the sphinx commands
.. to make sure the config-docs are available in the .rst format!
Supported datasets
------------------

If something that you need is not supported, feel free to
`raise an issue <https://github.com/NVIDIA/NeMo-speech-data-processor/issues>`_
or try to add the new processing yourself. Contributions from the community are always
welcome and encouraged!

The following datasets are already supported by SDP.

Mozilla Common Voice (MCV)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dataset link:** https://commonvoice.mozilla.org/en

**Required manual steps:** MCV requires agreeing to certain conditions, so you'd need to manually
download the data archive and specify its location with the ``raw_data_dir`` parameter of the
:ref:`CreateInitialManifestMCV <TODO>` class.

**Supported configs**.

* **Italian**:
  `config <https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/dataset_configs/italian/mcv/config.yaml>`_ |
  :doc:`documentation <config-docs/italian/mcv/config>`
* **Spanish**:
  `config <https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/dataset_configs/spanish_pc/mcv/config.yaml>`_ |
  :doc:`documentation <config-docs/spanish_pc/mcv12/config>`


Multilingual LibriSpeech (MLS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dataset link:** https://www.openslr.org/94/

**Supported configs**.

*Italian.*

.. include:: config-docs/italian/mls/config.rst

*Spanish (no punctuation and capitalization).*

.. include:: config-docs/spanish/mls/config.rst

*Spanish (with punctuation and capitalization).*

.. include:: config-docs/spanish_pc/mls/config.rst

VoxPopuli
~~~~~~~~~

**Dataset link:** https://github.com/facebookresearch/voxpopuli

**Supported configs**.

*Italian.*

.. include:: config-docs/italian/voxpopuli/config.rst

*Spanish.*

.. include:: config-docs/spanish_pc/voxpopuli/config.rst

Fisher
~~~~~~

**Dataset link:** https://catalog.ldc.upenn.edu/LDC2004T19

**Required manual steps:** You need to manually download the data from the above link.

**Supported configs**.

*Spanish.*

.. include:: config-docs/spanish_pc/fisher/config.rst


UK and Ireland English Dialect (SLR83)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dataset link:** https://openslr.org/83/

TBD


Corpus of Regional African American Language (CORAAL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dataset link:** https://oraal.uoregon.edu/coraal

TBD