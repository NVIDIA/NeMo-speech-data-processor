SDP architecture
----------------

TBD

General workflow
~~~~~~~~~~~~~~~~

TBD

SDP tests
~~~~~~~~~

It is important to make sure that your data processing code has the effect you intend, so SDP has a few different types of tests:

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

Unit/integration tests
######################

SDP also has a suit of unit/integration tests which can be run locally with
``python -m pytest tests/`` and will be run during the GitHub CI process. There are 2 sub-types:

a. `"End to end" tests <https://github.com/NVIDIA/NeMo-speech-data-processor/blob/main/tests/test_cfg_end_to_end_tests.py>`_
   which run SDP on a mini version of the raw initial dataset, and make sure the final manifest matches
   the reference final manifest.
b. `"Unit tests" for processors and utils <https://github.com/NVIDIA/NeMo-speech-data-processor/tree/main/tests>`_.
