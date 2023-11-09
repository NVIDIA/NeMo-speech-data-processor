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

"""
We do small-scale tests with small values of in_memory_chunksize to check that
processors work correctly even when chunking is used.
"""


import json

import pytest

from sdp.processors import DropNonAlphabet
from sdp.processors import SubMakeLowercase

def test_submakelowercase_with_chunking(tmp_path):

	input_lines = [
		{"text": "ABC"},
		{"text": "DEF"},
		{"text": "GHI"},
		{"text": "JKL"},
		{"text": "MNO"},
		{"text": "PQR"},
		{"text": "STU"},
		{"text": "VWX"},
		{"text": "YZ"},
	]

	expected_output_lines = [
		{"text": "abc"},
		{"text": "def"},
		{"text": "ghi"},
		{"text": "jkl"},
		{"text": "mno"},
		{"text": "pqr"},
		{"text": "stu"},
		{"text": "vwx"},
		{"text": "yz"},
	]


	# save input lines to manifest:
	input_manifest_file = tmp_path / "input_manifest.json"
	with open(input_manifest_file, "w") as f:
		for line in input_lines:
			f.write(json.dumps(line) + "\n")

	# run make_lowercase processor:
	output_manifest_file = tmp_path / "output_manifest_make_lowercase.json"
	processor = SubMakeLowercase(
		input_manifest_file=input_manifest_file,
		output_manifest_file=output_manifest_file,
		in_memory_chunksize=2
	)

	processor.process()

	# check that output manifest matches expected lines:
	with open(output_manifest_file, "r") as f:
		output_lines = [json.loads(line) for line in f]

	assert output_lines == expected_output_lines


def test_dropnonalphabet_with_chunking(tmp_path):

	input_lines = [
		{"text": "ABC"},
		{"text": "DEF"},
		{"text": "GHI"},
		{"text": "JKL"},
		{"text": "MNO"},
		{"text": "PQR"},
		{"text": "STU"},
		{"text": "VWX"},
		{"text": "YZ"},
	]

	expected_output_lines = [
		{"text": "ABC"},
		]

	# save input lines to manifest:
	input_manifest_file = tmp_path / "input_manifest.json"
	with open(input_manifest_file, "w") as f:
		for line in input_lines:
			f.write(json.dumps(line) + "\n")

	# run make_lowercase processor:
	output_manifest_file = tmp_path / "output_manifest_make_lowercase.json"
	processor = DropNonAlphabet(
		input_manifest_file=input_manifest_file,
		output_manifest_file=output_manifest_file,
		in_memory_chunksize=2,
		alphabet="ABC"
	)

	processor.process()

	# check that output manifest matches expected lines:
	with open(output_manifest_file, "r") as f:
		output_lines = [json.loads(line) for line in f]

	assert output_lines == expected_output_lines
