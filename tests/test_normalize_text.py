# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import pytest

from sdp.processors.modify_manifest.data_to_data import (
    InverseNormalizeText,
    NormalizeText,
)

normalize_test_params_list = []

normalize_test_params_list.extend(
    [
        (
            {
                "input_text_key": "text",
                "input_language": "en",
                "input_case": "cased",
                "output_text_key": "normalized_text",
            },
            {"text": "$12"},
            {"text": "$12", "normalized_text": "twelve dollars"},
        ),
        (
            {
                "input_text_key": "text",
                "input_language": "en",
                "input_case": "cased",
                "output_text_key": "normalized_text",
            },
            {"text": "120"},
            {"text": "120", "normalized_text": "one hundred and twenty"},
        ),
        (
            {
                "input_text_key": "text",
                "input_language": "hy",
                "input_case": "cased",
                "output_text_key": "normalized_text",
            },
            {"text": "11"},
            {"text": "11", "normalized_text": "տասնմեկ"},
        ),
    ]
)


@pytest.mark.parametrize("class_kwargs,test_input,expected_output", normalize_test_params_list, ids=str)
def test_normalize_text(class_kwargs, test_input, expected_output):
    processor = NormalizeText(**class_kwargs, output_manifest_file=None)
    processor.prepare()

    output = processor.process_dataset_entry(test_input)[0].data

    assert output == expected_output


inverse_normalize_test_params_list = []

inverse_normalize_test_params_list.extend(
    [
        (
            {
                "input_text_key": "text",
                "input_language": "en",
                "input_case": "cased",
                "output_text_key": "inverse_normalized_text",
            },
            {"text": "twelve dollars"},
            {"text": "twelve dollars", "inverse_normalized_text": "$12"},
        ),
        (
            {
                "input_text_key": "text",
                "input_language": "en",
                "input_case": "cased",
                "output_text_key": "inverse_normalized_text",
            },
            {"text": "one hundred and twenty"},
            {"text": "one hundred and twenty", "inverse_normalized_text": "120"},
        ),
        (
            {
                "input_text_key": "text",
                "input_language": "hy",
                "input_case": "cased",
                "output_text_key": "inverse_normalized_text",
            },
            {"text": "տասնմեկ"},
            {"text": "տասնմեկ", "inverse_normalized_text": "11"},
        ),
    ]
)


@pytest.mark.parametrize("class_kwargs,test_input,expected_output", inverse_normalize_test_params_list, ids=str)
def test_inverse_normalize_text(class_kwargs, test_input, expected_output):
    processor = InverseNormalizeText(**class_kwargs, output_manifest_file=None)
    processor.prepare()

    output = processor.process_dataset_entry(test_input)[0].data

    assert output == expected_output
