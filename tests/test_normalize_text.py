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

import pytest

from sdp.processors.modify_manifest.data_to_data import NormalizeText

test_params_list = []

test_params_list.extend(
    [
        (
            {
                "input_text_field": "text",
                "input_language": "en",
                "input_case": "cased",
                "output_text_field": "normalized_text",
            },
            {"text": "12$"},
            {"text": "12$", "normalized_text": "twelve dollar"},
        ),
        (
            {
                "input_text_field": "text",
                "input_language": "en",
                "input_case": "cased",
                "output_text_field": "normalized_text",
            },
            {"text": "120"},
            {"text": "120", "normalized_text": "one hundred and twenty"},
        ),
        (
            {
                "input_text_field": "text",
                "input_language": "hy",
                "input_case": "cased",
                "output_text_field": "normalized_text",
            },
            {"text": "11"},
            {"text": "11", "normalized_text": "տասնմեկ"},
        ),
    ]
)


@pytest.mark.parametrize(
    "class_kwargs,test_input,expected_output", test_params_list, ids=str
)
def test_normalize_text(class_kwargs, test_input, expected_output):
    processor = NormalizeText(**class_kwargs, output_manifest_file=None)
    processor.prepare()

    print(test_input)
    output = processor.process_dataset_entry(test_input)[0].data

    assert output == expected_output
