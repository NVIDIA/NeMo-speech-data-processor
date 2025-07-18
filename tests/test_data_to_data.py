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

from sdp.processors.modify_manifest.data_to_data import (
    InsIfASRInsertion,
    SubIfASRSubstitution,
    SubMakeLowercase,
    SubRegex,
)

from sdp.processors.inference.llm.post_processing.qwen_cleaning import CleanQwenGeneration

test_params_list = []

test_params_list.extend(
    [
        (
            InsIfASRInsertion,
            {"insert_words": [" nemo", "nemo ", " nemo "]},
            {"text": "i love the toolkit", "pred_text": "i love the nemo toolkit"},
            [{"text": "i love the nemo toolkit", "pred_text": "i love the nemo toolkit"}],
        ),
        (
            InsIfASRInsertion,
            {"insert_words": [" nemo", "nemo ", " nemo "]},
            {"text": "i love the toolkit", "pred_text": "i love the new nemo toolkit"},
            [{"text": "i love the toolkit", "pred_text": "i love the new nemo toolkit"}],
        ),
    ]
)

test_params_list.extend(
    [
        (
            SubIfASRSubstitution,
            {"sub_words": {"nmo ": "nemo "}},
            {"text": "i love the nmo toolkit", "pred_text": "i love the nemo toolkit"},
            [{"text": "i love the nemo toolkit", "pred_text": "i love the nemo toolkit"}],
        ),
    ]
)

test_params_list.extend(
    [
        (
            SubIfASRSubstitution,
            {"sub_words": {"nmo ": "nemo "}},
            {"text": "i love the nmo toolkit", "pred_text": "i love the nemo toolkit"},
            [{"text": "i love the nemo toolkit", "pred_text": "i love the nemo toolkit"}],
        ),
    ]
)

test_params_list.extend(
    [
        (
            SubMakeLowercase,
            {},
            {"text": "Hello Привет 123"},
            [{"text": "hello привет 123"}],
        ),
        (
            SubMakeLowercase,
            {"text_key": "text_new"},
            {"text_new": "Hello Привет 123"},
            [{"text_new": "hello привет 123"}],
        ),
    ]
)

test_params_list.extend(
    [
        (
            SubRegex,
            {"regex_params_list": [{"pattern": "\s<.*>\s", "repl": " "}]},
            {"text": "hello <cough> world"},
            [{"text": "hello world"}],
        ),
    ]
)

test_params_list.extend(
    [
        # Case: generation is fine, no replacement
        (
            CleanQwenGeneration,
            {"cer_threshold": 10, "upper_case_threshold": 0.6},
            {"text": "hello world", "generation": "hello world"},
            [{"text": "hello world", "generation": "hello world"}],
        ),

        # Case: generation is completely uppercase → replaced
        (
            CleanQwenGeneration,
            {"cer_threshold": 10, "upper_case_threshold": 0.5},
            {"text": "hello world", "generation": "HELLO WORLD"},
            [{"text": "hello world", "generation": "hello world"}],
        ),

        # Case: generation contains <|endoftext|> and prompt remnants → cleaned
        (
            CleanQwenGeneration,
            {},
            {"text": "hello", "generation": "Input transcript: hello\nOutput transcript: hello<|endoftext|>"},
            [{"text": "hello", "generation": "hello"}],
        ),

        # Case: generation is too different → high CER → replaced
        (
            CleanQwenGeneration,
            {"cer_threshold": 0.2},
            {"text": "hello world", "generation": "xyz abc"},
            [{"text": "hello world", "generation": "hello world"}],
        ),

        # Case: generation is empty → replaced
        (
            CleanQwenGeneration,
            {},
            {"text": "reference", "generation": ""},
            [{"text": "reference", "generation": "reference"}],
        ),

        # Case: text is empty → fallback to replacement
        (
            CleanQwenGeneration,
            {},
            {"text": "", "generation": "some output"},
            [{"text": "", "generation": ""}],
        ),
    ]
)

@pytest.mark.parametrize("test_class,class_kwargs,test_input,expected_output", test_params_list, ids=str)
def test_data_to_data(test_class, class_kwargs, test_input, expected_output):
    processor = test_class(**class_kwargs, output_manifest_file=None)
    result = [entry.data for entry in processor.process_dataset_entry(test_input)]

    assert result == expected_output