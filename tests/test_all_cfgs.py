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
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from nemo.utils import logging
from sdp.run_processors import run_processors


CONFIG_BASE_DIR = Path(__file__).parents[1] / "dataset_configs"


def get_test_cases():
    """Returns paths to all configs that are checked in."""
    for config_path in glob.glob(f"{CONFIG_BASE_DIR}/**/*.yaml", recursive=True):
        yield config_path


@pytest.mark.skipif(
    not os.getenv("TEST_DATA_ROOT"), reason="TEST_DATA_ROOT has to be defined for e2e config tests"
)
@pytest.mark.parametrize("config_path", get_test_cases())
def test_configs(config_path, tmp_path):
    TEST_DATA_ROOT = os.environ["TEST_DATA_ROOT"]
    # we expect CONFIG_DIR and TEST_DATA_ROOT to have the same structure (e.g. <lang>/<dataset>)
    rel_path_from_root = os.path.relpath(Path(config_path).parent, CONFIG_BASE_DIR)
    reference_manifest = str(
        Path(TEST_DATA_ROOT) / rel_path_from_root / "test_data_reference.json"
    )
    if not os.path.exists(reference_manifest):
        raise ValueError(
            f"No such file {reference_manifest}. Are you sure you specified the "
            " 'TEST_DATA_ROOT' environment variable correctly? "
            "We expect CONFIG_DIR and TEST_DATA_ROOT to have the same "
            "structure (e.g. <lang>/<dataset>)"
        )
    cfg = OmegaConf.load(config_path)
    assert "processors" in cfg
    cfg["processors_to_run"] = "all"
    cfg["workspace_dir"] = str(tmp_path)
    cfg["final_manifest"] = str(tmp_path / "final_manifest.json")
    cfg["data_split"] = "train"
    cfg["processors"][0]["use_test_data"] = True

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
