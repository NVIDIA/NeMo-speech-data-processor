How to add a new processor?
---------------------------

We will describe how to make your own processor classes by referring to SDP's existing classes.

Creating an initial manifest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One of the child classes of ``BaseParallelProcessor`` provided in SDP is ``CreateInitialManifestMLS``. It downloads raw MLS data for a specified language, and creates an initial manifest (in the format expected by NeMo) which can be cleaned by subsequent processors.

The ``CreateInitialManifestMLS.prepare()`` method downloads and extracts the raw data.

The ``CreateInitialManifestMLS.read_manifest()`` method reads the lines in the raw MLS transcript file.

The ``CreateInitialManifestMLS.process_dataset_entry()`` method takes in the lines from the raw MLS transcript file, and outputs ``DataEntry`` objects containing entries that will be saved into the manifest (i.e. ``"audio_filepath"``, ``"duration"``, ``"text"``) for each utterance.


A **ModifyManifestTextProcessor** subclass that cleans the reference text
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the classes provided in SDP is ``SubRegex``. At initialization, it takes in ``regex_params_list``, a list of dictionaries which must contain the keys ``"pattern"``, ``"repl"``, and, optionally, ``"count"``. These keys will be used to apply regex substitutions using these parameters fed into ``re.sub``. The substitutions will be applied to the data at ``text_key`` (i.e. ``data_entry.data[self.text_key]``). By default, ``text_key="text"``, i.e. the substitutions will be applied to the ``"text"`` attribute of the manifest.

In its ``_process_dataset_entry(data_entry)`` method, the ``SubRegex`` processor does the string to string conversion upon the ``data_entry`` that is input. Its output is a ``data_entry`` with the changes applied to ``data``, and the the metrics of which regex patterns caused a substitution to be made. These metrics will be aggregated over all utterances by the ``BaseParallelProcessor`` class. ``SubRegex`` also has a ``finalize(metrics)`` method which will log information about the aggregated metrics after all of the utterances in the manifest have been processed.

A **ModifyManifestTextProcessor** subclass that drops incorrectly transcribed utterances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the classes provided in SDP is ``DropHighLowCharrate``. At initialization, it takes in ``high_charrate_threshold`` and ``low_charrate_threshold``, for which the utterance will be dropped if it is above or below each value respectively. This is helpful for automatically filtering out incorrectly transcribed utterances.

In its ``_process_dataset_entry(data_entry)`` method it evaluates the character rate of the utterance(by dividing the length of ``data_entry.data[self.text_key]`` by the value of ``data_entry.data["duration"]``). If the character rate is within bounds, it will return the same ``data_entry`` that was input. If the character rate is out of bounds, it will return a ``data_entry`` with ``data=None`` and ``metrics`` which reflect the applied changes.
Similar to the ``SubSubstringToSpace`` class, it has a ``finalize(metrics)`` method which will log information about the aggregated metrics after all of the utterances in the manifest have been processed.

Class diagram
-------------
A diagram of the classes mentioned above is included here. Arrows represent inheritance.

We omit the details of the ``CreateInitialManifestMLS`` class in the diagram in order to save space.


.. raw:: html

    <div align="center">
      <img src="https://mermaid.ink/img/pako:eNqlVMFu2zAM_ZVApw1o8wHBLl17WIEGGOYCuxgQWImOhcqSQdFtM6__PjmSvbhzsgI1fKDI98gnklAvlNcoNkJZCOHGwI6gKd1XCPidvMIQPK2-_L68XB1cQGAt2im0iLwqfty6CgmdwkXATzKMW3CmwsCly5i3uRP2mhAYb51hA3bkbO-Ks6St16baj-h7fOEjxaU7E078onuIf2AybnfvixaGi_yXdUO-_WZ29Z1_vq6BKOoeqh06u5q1oS_dKn6-47Zj2eSUsjIWU8S4E4E2pfj0OR05Rgf7dVbmbVP6RW5L2ALheIx91lPFv5gDRWrgmJglOqb9GKyMA2t-4UzA8fCnusgExmHMH5fNJu8DsKpliPx_1E3JZovSj1XR6iDZywBPZ7inFienWa_Xk7GeEc_MuR-7_sLyEffT9bScu4axSBU7FuZjOt3S4ZTMDJPvwE2SF_Y1Sw2jO7w_7Wy2TZydUeG42sKe52p19EqVfZJrwlB7q1PQ-ueTsQ_IisLEhWiQGjA6PmQHKaXgGhssxSaaGivoLJciQaFjX-ydEpsKbMAL0bWxDua3L3tf_wDMstkP" height=100% />
    </div>
