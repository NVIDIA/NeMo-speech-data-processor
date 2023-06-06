# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from pathlib import Path

import hydra
from omegaconf import OmegaConf, open_dict
import pytest


DATASET_CONFIGS_ROOT = Path(__file__).parents[1] / "dataset_configs"


def get_test_cases():
    """Returns paths to all configs that are checked in."""
    for config_path in glob.glob(f"{DATASET_CONFIGS_ROOT}/**/*.yaml", recursive=True):
        yield config_path


@pytest.mark.parametrize("config_path", get_test_cases())
def test_configs(config_path: str):
    try:
        # to be able to correctly read some of the configs
        OmegaConf.register_new_resolver("subfield", lambda node, field: node[field])
    except ValueError:  # already registered
        pass

    cfg = OmegaConf.load(config_path)
    cfg["data_split"] = "train"  # in case it's required for tests
    for processor_cfg in cfg.processors:
        if "test_cases" in processor_cfg:
            # clear input_manifest_file and output_manifest_file to make sure we don't get
            # a MissingMandatoryValue error when we instantiate the processor
            with open_dict(processor_cfg):
                processor_cfg["output_manifest_file"] = None
                processor_cfg["input_manifest_file"] = None
            processor = hydra.utils.instantiate(processor_cfg)
            processor.test()
