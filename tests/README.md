To run tests you will need to install additional packages from
[tests/requirements.txt](requirements.txt).

Command to run all tests: `python -m pytest tests/`.

There are multiple levels of tests that we use:

- unit tests and doc tests for various SDP components.
- full end-to-end tests that will run all configs from "dataset_configs" folder
  on the small subset of datasets. Require defining `TEST_DATA_ROOT` environment
  variable with a path to the test data.  Note that if `TEST_DATA_ROOT`
  is not defined, e2e tests are skipped. 

### For SDP maintainers - how to set up end-to-end tests for a dataset.
Once you are happy with the config & code for a dataset, you can also set up an end-to-end test for it to make sure that future changes to SDP will not affect the workings of your config & code.

The steps for this are as follows:

1. Create a script like `tests/prepare_test_data/prepare_mls_data.py` which you will use to make a mini version of the initial dataset that is read by the first SDP processor for your dataset. Run this script.
2. Extend the functionality of the first SDP processor for the dataset such that it will read the mini initial dataset (not the full initial dataseet) if the `use_test_data` flag is `True`. This will probably entail adding an if-else branch in the `.prepare()` method of that processor class. 
3. Run the SDP dataset creation process for your dataset but with flags like `data_split=True, final_manifest=test_data_reference.json, processor.0.use_test_data=True, workspace_dir=<some empty directory which you may delete after this step>`.
4. Save the mini initial dataset produced in step 1, and the final manifest produced in step 3 in the location of `<TEST_DATA_ROOT>/<language>/<dataset>`.
    
    a. If you save them locally, the end-to-end test will work locally.

    b. If you save them in the SDP tests AWS S3 bucket (you can only do this if you have access), the tests will be able to work when the Github CI is run.


