# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import json
import os
from functools import partial
from pathlib import Path
from typing import Callable, Dict

import pytest
from omegaconf import OmegaConf

from sdp.logging import logger
from sdp.run_processors import run_processors

DATASET_CONFIGS_ROOT = Path(__file__).parents[1] / "dataset_configs"


def data_check_fn_mls(raw_data_dir: str, language: str) -> None:
    """Raises error if do not find expected data"""
    expected_file = Path(raw_data_dir) / f"mls_{language}.tar.gz"
    if not expected_file.exists():
        raise ValueError(f"No such file {str(expected_file)}")


def data_check_fn_mcv(raw_data_dir: str, archive_file_stem: str) -> None:
    """Raises error if do not find expected data"""
    expected_file = Path(raw_data_dir) / f"{archive_file_stem}.tar.gz"
    if not expected_file.exists():
        raise ValueError(f"No such file {str(expected_file)}")


def get_test_cases():
    """Returns paths, and data check fn for all configs that we want to test."""

    return [
        (f"{DATASET_CONFIGS_ROOT}/spanish/mls/config.yaml", partial(data_check_fn_mls, language="spanish")),
        (
            f"{DATASET_CONFIGS_ROOT}/spanish_pc/mcv12/config.yaml",
            partial(data_check_fn_mcv, archive_file_stem="cv-corpus-12.0-2022-12-07-es"),
        ),
    ]


def check_e2e_test_data() -> bool:
    """Checks if required environment variables are defined for e2e data.

    Either TEST_DATA_ROOT needs to be defined or both AWS_SECRET_KEY
    and AWS_ACCESS_KEY.
    """
    if os.getenv("TEST_DATA_ROOT"):
        return True
    if os.getenv("AWS_SECRET_KEY") and os.getenv("AWS_ACCESS_KEY"):
        return True
    return False


def get_e2e_test_data_path() -> str:
    """Returns path to e2e test data (downloading from AWS if necessary).

    In case of downloading from AWS, will create "test_data" folder in the
    current folder and set TEST_DATA_ROOT automatically (used by the sdp code
    to locate test data).
    """
    test_data_root = os.getenv("TEST_DATA_ROOT")
    if test_data_root:  # assume it's present locally
        return test_data_root

    import boto3

    s3_resource = boto3.resource(
        "s3", aws_access_key_id=os.getenv("AWS_ACCESS_KEY"), aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    )
    bucket = s3_resource.Bucket("sdp-test-data")
    for obj in bucket.objects.all():
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))
        bucket.download_file(obj.key, obj.key)

    os.environ["TEST_DATA_ROOT"] = os.path.abspath("test_data")

    return os.environ["TEST_DATA_ROOT"]


@pytest.mark.skipif(
    not check_e2e_test_data(),
    reason="Either TEST_DATA_ROOT needs to be defined or both AWS_SECRET_KEY "
    "and AWS_ACCESS_KEY to run e2e config tests",
)
@pytest.mark.parametrize("config_path,data_check_fn", get_test_cases())
def test_configs(config_path: str, data_check_fn: Callable, tmp_path: str):

    test_data_root = get_e2e_test_data_path()
    # we expect DATASET_CONFIGS_ROOT and TEST_DATA_ROOT
    # to have the same structure (e.g. <lang>/<dataset>)
    rel_path_from_root = os.path.relpath(Path(config_path).parent, DATASET_CONFIGS_ROOT)

    # run data_check_fn - it will raise error if the expected test data is not found
    data_check_fn(raw_data_dir=str(Path(test_data_root) / rel_path_from_root))

    reference_manifest = str(Path(test_data_root) / rel_path_from_root / "test_data_reference.json")
    if not os.path.exists(reference_manifest):
        raise ValueError(f"Did not find reference manifest {reference_manifest}")

    cfg = OmegaConf.load(config_path)
    assert "processors" in cfg
    cfg["processors_to_run"] = "all"
    cfg["workspace_dir"] = str(tmp_path)
    cfg["final_manifest"] = str(tmp_path / "final_manifest.json")
    cfg["data_split"] = "train"
    cfg["processors"][0]["raw_data_dir"] = str(Path(test_data_root) / rel_path_from_root)

    run_processors(cfg)
    # additionally, let's test that final generated manifest matches the
    # reference file (ignoring the file paths)
    with open(reference_manifest, "rt", encoding="utf8") as reference_fin, open(
        cfg["final_manifest"], "rt", encoding="utf8"
    ) as generated_fin:
        reference_lines = reference_fin.readlines()
        generated_lines = generated_fin.readlines()
        assert len(reference_lines) == len(generated_lines)
        for reference_line, generated_line in zip(reference_lines, generated_lines):
            reference_data = json.loads(reference_line)
            generated_data = json.loads(generated_line)
            reference_data.pop("audio_filepath")
            generated_data.pop("audio_filepath")
            assert reference_data == generated_data
