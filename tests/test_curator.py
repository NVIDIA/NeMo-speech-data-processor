# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
import tempfile
from pathlib import Path

import yaml
from omegaconf import OmegaConf
from ray_curator.stages.base import ProcessingStage
from ray_curator.tasks import Task, _EmptyTask

from sdp.run_processors import run_processors


def _write_config(file_path: Path, dict_conf):
    with file_path.open("w") as file:
        yaml.dump(dict_conf, file)


def _make_dict(output_manifest_file, use_backend=None):
    workspace_dir = os.path.join(os.getenv('TEST_DATA_ROOT'), "armenian/audio_books/mp3")
    return {
        "processors_to_run": "0:",
        "use_backend": use_backend,
        "processors": [
            {
                "_target_": "sdp.processors.CreateInitialManifestByExt",
                "raw_data_dir": workspace_dir,
                "extension": "mp3",
                "output_file_key": "audio_filepath",
                "output_manifest_file": output_manifest_file,
            },
        ],
    }


def _make_expected_output():
    workspace_dir = os.path.join(os.getenv('TEST_DATA_ROOT'), "armenian/audio_books/mp3")
    return {'audio_filepath': os.path.join(workspace_dir, "Eleonora/Eleonora30s.mp3")}


def test_curator():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "output_manifest_file.jsonl")
        dict_conf = _make_dict(output_manifest_file=output_path, use_backend="curator")
        conf_path = Path(tmpdir) / "config.yaml"
        _write_config(conf_path, dict_conf)

        cfg = OmegaConf.load(conf_path)

    run_processors(cfg)
    with open(output_path, "r") as f:
        output = json.load(f)

    expected_output = _make_expected_output()
    assert output == expected_output, f"Expected {expected_output}, but got {output}"
