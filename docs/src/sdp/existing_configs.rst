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

**Dataset link:** https://commonvoice.mozilla.org/

**Required manual steps:** MCV requires agreeing to certain conditions, so you'd need to manually
download the data archive and specify its location with the ``raw_data_dir`` parameter of the
:class:`sdp.processors.CreateInitialManifestMCV` class.

**Supported configs**.

* **Italian**:
  `config <https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/dataset_configs/italian/mcv/config.yaml>`__ |
  :doc:`documentation <config-docs/italian/mcv/config>`
* **Spanish**:
  `config <https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/dataset_configs/spanish_pc/mcv12/config.yaml>`__ |
  :doc:`documentation <config-docs/spanish_pc/mcv12/config>`

.. toctree::
   :hidden:

   config-docs/italian/mcv/config
   config-docs/spanish_pc/mcv12/config

Multilingual LibriSpeech (MLS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dataset link:** https://www.openslr.org/94/

**Supported configs**.

* **Italian (with punctuation and capitalization)**:
  `config <https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/dataset_configs/italian/mls/config.yaml>`__ |
  :doc:`documentation <config-docs/italian/mls/config>`
* **Italian (no punctuation and capitalization)**:
  `config <https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/dataset_configs/italian/mls/config_nopc.yaml>`__ |
  :doc:`documentation <config-docs/italian/mls/config_nopc>`
* **Spanish (with punctuation and capitalization)**:
  `config <https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/dataset_configs/spanish_pc/mls/config.yaml>`__ |
  :doc:`documentation <config-docs/spanish_pc/mls/config>`
* **Spanish (no punctuation and capitalization)**:
  `config <https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/dataset_configs/spanish/mls/config.yaml>`__ |
  :doc:`documentation <config-docs/spanish/mls/config>`

.. toctree::
   :hidden:

   config-docs/italian/mls/config
   config-docs/italian/mls/config_nopc
   config-docs/spanish_pc/mls/config
   config-docs/spanish/mls/config

VoxPopuli
~~~~~~~~~

**Dataset link:** https://github.com/facebookresearch/voxpopuli

**Supported configs**.

* **Italian**:
  `config <https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/dataset_configs/italian/voxpopuli/config.yaml>`__ |
  :doc:`documentation <config-docs/italian/voxpopuli/config>`
* **Spanish**:
  `config <https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/dataset_configs/spanish_pc/voxpopuli/config.yaml>`__ |
  :doc:`documentation <config-docs/spanish_pc/voxpopuli/config>`

.. toctree::
   :hidden:

   config-docs/italian/voxpopuli/config
   config-docs/spanish_pc/voxpopuli/config

Fisher
~~~~~~

**Dataset link:** https://catalog.ldc.upenn.edu/LDC2004T19

**Required manual steps:** You need to manually download the data from the above link.

**Supported configs**.

* **Spanish**:
  `config <https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/dataset_configs/spanish_pc/fisher/config.yaml>`__ |
  :doc:`documentation <config-docs/spanish_pc/fisher/config>`

.. toctree::
   :hidden:

   config-docs/spanish_pc/fisher/config

UK and Ireland English Dialect (SLR83)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dataset link:** https://openslr.org/83/

`config <https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/dataset_configs/english/slr83/config.yaml>`__ |
:doc:`documentation <config-docs/english/slr83/config>`

.. toctree::
   :hidden:

   config-docs/english/slr83/config

Corpus of Regional African American Language (CORAAL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dataset link:** https://oraal.uoregon.edu/coraal

`config <https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/dataset_configs/english/coraal/config.yaml>`__ |
:doc:`documentation <config-docs/english/coraal/config>`

.. toctree::
   :hidden:

   config-docs/english/coraal/config

Corpus of Armenian Text to Upload into Common Voice (MCV)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dataset link:** https://commonvoice.mozilla.org/

`config <https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/dataset_configs/armenian/text_mcv/config.yaml>`__ |
:doc:`documentation <config-docs/armenian/text_mcv/config>`

.. toctree::
   :hidden:

   config-docs/armenian/text_mcv/config

Corpus based on Armenian audiobooks 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


`config <https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/dataset_configs/armenian/audio_books/config.yaml>`__ |
:doc:`documentation <config-docs/armenian/audio_books/config>`

.. toctree::
   :hidden:

   config-docs/armenian/audio_books/config