To run tests you will need to install additional packages from
[tests/requirements.txt](requirements.txt).

Command to run all tests: `pytest`.

There are multiple levels of tests that we use:

- unit tests and doc tests for various SDP components.
- full end-to-end tests that will run all configs from "dataset_configs" folder
  on the small subset of datasets. Require defining `TEST_DATA_ROOT` environment
  variable with a path to the test data.  Note that if `TEST_DATA_ROOT`
  is not defined, e2e tests are skipped. TODO: add more details on how to
  generate the data and expected structure.
