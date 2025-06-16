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

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Union

import pytest

from sdp.processors import ApplyInnerJoin, DropNonAlphabet


def _write_manifest(manifest: Path, entries: List[Dict[str, Union[str, float]]]):
    with open(manifest, "w") as f:
        for line in entries:
            f.write(json.dumps(line) + "\n")


def test_empty_test_cases():
    """Testing that empty test cases don't raise an error."""
    processor = DropNonAlphabet("123", output_manifest_file="tmp")
    processor.test()


inner_join_entries = []
inner_join_entries.extend(
    [
        (
            [
                {"id": 1, "text": "abc1", "duration": 10, "audio_filepath": "path1"},
                {"id": 2, "text": "abc2", "duration": 11, "audio_filepath": "path2"},
                {"id": 3, "text": "abc3", "duration": 11, "audio_filepath": "path3"},
            ],
            [
                {"id": 1, "text": "abc1", "duration": 10, "audio_filepath": "path1"},
                {"id": 2, "text": "abc3", "duration": 11, "audio_filepath": "path_2"},
            ],
            [
                {"id": 1, "text": "abc1", "duration": 10, "audio_filepath": "path1"},
            ],
            None,
        ),
        (
            [
                {"id": 1, "text": "abc1", "duration": 10, "audio_filepath": "path1"},
                {"id": 2, "text": "abc2", "duration": 11, "audio_filepath": "path2"},
                {"id": 3, "text": "abc3", "duration": 11, "audio_filepath": "path3"},
            ],
            [
                {"id": 1, "text": "abc1", "duration": 10, "audio_filepath": "path1"},
                {"id": 2, "text": "abc2", "duration": 11, "audio_filepath": "path2"},
            ],
            [
                {"id": 1, "text": "abc1", "duration": 10, "audio_filepath": "path1"},
                {"id": 2, "text": "abc2", "duration": 11, "audio_filepath": "path2"},
            ],
            None,
        ),
        (
            [
                {"id": 1, "text": "abc1", "duration": 10, "audio_filepath": "path1"},
                {"id": 2, "text": "abc2", "duration": 11, "audio_filepath": "path2"},
                {"id": 3, "text": "abc3", "duration": 11, "audio_filepath": "path3"},
            ],
            [
                {"id": 1, "text": "abc1", "duration": 10, "audio_filepath": "path1"},
                {"id": 2, "text": "abc_2", "duration": 11, "audio_filepath": "path2"},
            ],
            [
                {
                    "id": 1,
                    "text_x": "abc1",
                    "duration_x": 10,
                    "audio_filepath_x": "path1",
                    "text_y": "abc1",
                    "duration_y": 10,
                    "audio_filepath_y": "path1",
                },
                {
                    "id": 2,
                    "text_x": "abc2",
                    "duration_x": 11,
                    "audio_filepath_x": "path2",
                    "text_y": "abc_2",
                    "duration_y": 11,
                    "audio_filepath_y": "path2",
                },
            ],
            "id",
        ),
        (
            [
                {"id": 1, "text": "abc1", "duration": 10, "audio_filepath": "path1"},
                {"id": 2, "text": "abc2", "duration": 11, "audio_filepath": "path2"},
                {"id": 3, "text": "abc3", "duration": 11, "audio_filepath": "path3"},
            ],
            [
                {"id": 1, "text": "abc1", "duration": 10, "audio_filepath": "path1"},
                {"id": 2, "text": "abc_2", "duration": 11, "audio_filepath": "path2"},
            ],
            [
                {
                    "id": 1,
                    "text_x": "abc1",
                    "duration": 10,
                    "audio_filepath": "path1",
                    "text_y": "abc1",
                },
                {
                    "id": 2,
                    "text_x": "abc2",
                    "duration": 11,
                    "audio_filepath": "path2",
                    "text_y": "abc_2",
                },
            ],
            ["id", "duration", "audio_filepath"],
        ),
        (
            [{"id": 1, "text": "text1"}],
            [{"id": 1, "text": "text2"}],
            [{"id": 1, "text_x": "text1", "text_y": "text2"}],
            "id",
        ),
    ]
)


@pytest.mark.parametrize("input1,input2,output,coloumn_id", inner_join_entries)
def test_apply_inner_join(
    input1: List[Dict[str, Union[str, float]]],
    input2: List[Dict[str, Union[str, float]]],
    output: List[Dict[str, Union[str, float]]],
    coloumn_id: Union[str, List[str], None],
):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        manifest1 = tmpdir_path / "manifest1.json"
        manifest2 = tmpdir_path / "manifest2.json"
        manifest_out = tmpdir_path / "output_manifest.json"

        _write_manifest(manifest1, input1)
        _write_manifest(manifest2, input2)

        processor = ApplyInnerJoin(
            left_manifest_file=manifest1,
            right_manifest_file=manifest2,
            column_id=coloumn_id,
            output_manifest_file=manifest_out,
        )

        processor.process()

        with open(manifest_out, "r") as f:
            output_lines = [json.loads(line) for line in f]

        assert output_lines == output
