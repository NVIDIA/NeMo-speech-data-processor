How to add a new processor?
---------------------------

.. TODO: maybe better to have a fully self-contained example with just some synthetic data?

We will describe how to make your own processor classes by referring to SDP's existing classes.

To understand this section better, it might be useful to skim through the description of the
:ref:`SDP's base classes <sdp-base-classes>`.

Creating an initial manifest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One of the child classes of :class:`sdp.processors.base_processor.BaseParallelProcessor` provided in SDP is
:class:`sdp.processors.CreateInitialManifestMLS`.

.. autoclass:: sdp.processors.CreateInitialManifestMLS
   :show-inheritance:
   :member-order: bysource
   :no-inherited-members:

It downloads raw MLS data for a specified language, and creates an initial manifest
(in the format expected by NeMo) which can be cleaned by subsequent processors.

The :meth:`sdp.processors.CreateInitialManifestMLS.prepare` method downloads and extracts the raw data.

The :meth:`sdp.processors.CreateInitialManifestMLS.read_manifest` method reads the lines in the raw MLS transcript file.

The :meth:`sdp.processors.CreateInitialManifestMLS.process_dataset_entry` method takes in the lines from the raw MLS
transcript file, and outputs ``DataEntry`` objects containing entries that will be saved into the
manifest (i.e. ``audio_filepath``, ``duration``, ``text``) for each utterance.


Cleaning the reference text
~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the classes provided in SDP is :class:`sdp.processors.SubRegex`.

.. autoclass:: sdp.processors.SubRegex
   :show-inheritance:
   :private-members:
   :member-order: bysource
   :no-inherited-members:
   :exclude-members: _abc_impl

At initialization, it takes in ``regex_params_list``, a list of dictionaries which must contain the
keys ``pattern``, ``repl``, and, optionally, ``count``.
These keys will be used to apply regex substitutions using these parameters fed into
``re.sub``. The substitutions will be applied to the data at ``text_key``
(i.e. ``data_entry.data[self.text_key]``). By default, ``text_key="text"``, i.e. the substitutions
will be applied to the ``"text"`` attribute of the manifest.

In its :meth:`sdp.processors.SubRegex.process_dataset_entry` method, the
processor does the string to string conversion upon the ``data_entry`` that is input.
Its output is a ``data_entry`` with the changes applied to ``data``, and the the metrics of
which regex patterns caused a substitution to be made.
These metrics will be aggregated over all utterances by the
:class:`sdp.processors.base_processor.BaseParallelProcessor` class.
:class:`sdp.processors.SubRegex` also has a :meth:`sdp.processors.SubRegex.finalize` method which will log
information about the aggregated metrics after all of the utterances in the manifest have been processed.

Filtering incorrect transcriptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of the classes provided in SDP is :class:`sdp.processors.DropHighLowCharrate`.

.. autoclass:: sdp.processors.DropHighLowCharrate
   :show-inheritance:
   :private-members:
   :member-order: bysource
   :no-inherited-members:
   :exclude-members: _abc_impl

At initialization, it takes in ``high_charrate_threshold`` and ``low_charrate_threshold``,
for which the utterance will be dropped if it is above or below each value respectively.
This is helpful for automatically filtering out incorrectly transcribed utterances.

In its :meth:`sdp.processors.DropHighLowCharrate.process_dataset_entry` method it evaluates the character rate of
the utterance(by dividing the length of ``data_entry.data[self.text_key]`` by the value of
``data_entry.data["duration"]``). If the character rate is within bounds, it will return the
same ``data_entry`` that was input. If the character rate is out of bounds, it will return
a ``data_entry`` with ``data=None`` and ``metrics`` which reflect the applied changes.
Similar to the :class:`sdp.processors.SubRegex` class, it has a :meth:`sdp.processors.DropHighLowCharrate.finalize`
method which will log information about the aggregated metrics after all of the utterances
in the manifest have been processed.

Class diagram
~~~~~~~~~~~~~
A diagram of the classes mentioned above is included here. Arrows represent inheritance.

.. raw:: html

    <div align="center">
      <img src="https://mermaid.ink/img/pako:eNqtVEtvm0AQ_itoT45kLLW9kV5a55BKrRSFKxKawOBdZdlFu0Ntl_q_d3iYYgesSC0X5v3NfDPaRmQ2RxGJTIP3Dwp2DsrEfAWPT85m6L11weffYRh0JnCgNerRNRv5JX7-Zgp0aDKcDXjaTvxDxHXlPjKuX55xh4ebQQ_OVo9qJ7_b_VaCc0B4M74Ec7Qk0VXO_lQ55t3s6NteEtMpwUXXTWIC_mxNVU0pp6sCPaWF0th7lFlwVH2J1V2vEntb-XQBc91lM5tbOazA4Vl1CPmI-DemS0lzIC5MKRpyx1Wr9eIQVigDWv3CVYnkVOaXupsucmyKU0AZZHA-HN1bX4AymXou-ab1sdhk6c2UNcJDyxjq_ILla_ME2EA54Gw2m2XA8-00Z8JYTplCKH2qlad_pWwEmrm_AVOyNc0GW0rSoZf2PJG2-0Xff-hq6cqbeepuQZ3EWpToSlA5vxRdgURwaV6DiFjMsYBaUyLWXcnBt43j3r0xfCmRIRnaIqRjhasPd7yOjIKGJ9A6CvZSEd4Hnpx9xXCvcpLRp-oQnO7f5n58b24iOjYSw91DTTY-mkxEBWiPa1FXPCEO791oxVyRdT-GF7H9nf4AezbWWg" height=100% />
    </div>
