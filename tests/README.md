To run tests you will need to install additional packages from
[tests/requirements.txt](requirements.txt).

Command to run all tests: `python -m pytest tests/`.

There are multiple levels of tests that we use:

- full end-to-end tests that will try to run all configs inside the `dataset_configs/` folder using small subsets of datasets. These tests require `TEST_DATA_ROOT` to be defined, either as an environment variable, or by accessing the AWS S3 bucket (which is used during Github CI tests). If `TEST_DATA_ROOT` is not defined, these end-to-end tests are skipped. These tests are run by the `tests/test_cfg_end_to_end_tests.py` file. These tests also require the processor that creates the initial manifest to have a `raw_data_dir` parameter.
- unit tests and doc tests for various SDP components.

### For SDP maintainers - how to set up end-to-end tests for a dataset.
Once you are happy with the config & code for a dataset, you can also set up an end-to-end test for it to make sure that future changes to SDP will not affect the workings of your config & code.

The steps for this are as follows:

1. Create a script like `tests/prepare_test_data/prepare_mls_data.py` which you will use to make a (archived) mini version of the initial dataset that is read by the first SDP processor for your dataset. Run this script.
2. Extend the functionality of the first SDP processor for the dataset such that if the dir containing the archived file outputed in step 1 is passed in as the `raw_data_dir` parameter to that processor, it will be able to extract and use the data in that archive for the rest of the data processing. This will probably entail adding some if-else branches in the `.prepare()` method of that processor class. Make sure the processor class will also work normally in non-testing mode.
3. Run the SDP dataset creation process for your dataset but with flags like `data_split=True, final_manifest=test_data_reference.json, processor.0.raw_data_dir=<path to the archived mini initial dataset created in step 1>, workspace_dir=<some empty directory which you may delete after this step>`.
4. Save the mini initial dataset produced in step 1, and the final manifest produced in step 3 in the location of `<TEST_DATA_ROOT>/<language>/<dataset>`.
    
    a. If you save the files locally, the end-to-end test will work locally.

    b. If you save the files in the SDP tests AWS S3 bucket (you can only do this if you have access), the tests will be able to work when the Github CI is run.


