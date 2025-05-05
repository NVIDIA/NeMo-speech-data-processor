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
    ListToEntries
)

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
        # Test: list of dictionaries (e.g., segments)
        (
        ListToEntries,
            {"field_with_list": "segments", "fields_to_remove": ["duration"]},
            {"audio_filepath": "a.wav", "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}, {"start": 1.1, "end": 2.0, "text": "World"}], "duration": 2.5},
            [{"audio_filepath": "a.wav", "start": 0.0, "end": 1.0, "text": "Hello"}, {"audio_filepath": "a.wav", "start": 1.1, "end": 2.0, "text": "World"}]
        ),
        # Test: list of primitive values (strings), requires output_field
        (
            ListToEntries,
            {"field_with_list": "text_chunks", "output_field": "text"},
            {"audio_filepath": "b.wav", "text_chunks": ["Привет", "Мир"], "lang": "ru"},
            [{"audio_filepath": "b.wav", "lang": "ru", "text": "Привет"}, {"audio_filepath": "b.wav", "lang": "ru", "text": "Мир"}]
        ),
        # Test: only keep specified fields (fields_to_save)
        (
            ListToEntries,
            {"field_with_list": "segments", "fields_to_save": ["audio_filepath"]},
            {"audio_filepath": "c.wav", "segments": [{"start": 0, "text": "A"}, {"start": 1, "text": "B"}], "remove_me": "to_delete"},
            [{"audio_filepath": "c.wav", "start": 0, "text": "A"}, {"audio_filepath": "c.wav", "start": 1, "text": "B"}]
        )
    ]
)


@pytest.mark.parametrize("test_class,class_kwargs,test_input,expected_output", test_params_list, ids=str)
def test_data_to_data(test_class, class_kwargs, test_input, expected_output):
    processor = test_class(**class_kwargs, output_manifest_file=None)
    result = [entry.data for entry in processor.process_dataset_entry(test_input)]

    assert result == expected_output
