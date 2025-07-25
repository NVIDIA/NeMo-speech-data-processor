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
    ListToEntries,
    LambdaExpression,
)

from sdp.processors.inference.llm.utils.qwen_cleaning import CleanQwenGeneration
from sdp.processors.inference.asr.utils.whisper_hallucinations import DetectWhisperHallucinationFeatures

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
            {"regex_params_list": [{"pattern": r"\s<.*>\s", "repl": " "}]},
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
            {"field_with_list": "segments"},
            {"audio_filepath": "a.wav", "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}, {"start": 1.1, "end": 2.0, "text": "World"}], "duration": 2.5},
            [{"audio_filepath": "a.wav", "duration": 2.5, "start": 0.0, "end": 1.0, "text": "Hello"}, {"audio_filepath": "a.wav", "duration": 2.5, "start": 1.1, "end": 2.0, "text": "World"}]
        ),
        # Test: list of primitive values (strings), requires output_field
        (
            ListToEntries,
            {"field_with_list": "text_chunks", "output_field": "text"},
            {"audio_filepath": "b.wav", "text_chunks": ["Привет", "Мир"], "lang": "ru"},
            [{"audio_filepath": "b.wav", "lang": "ru", "text": "Привет"}, {"audio_filepath": "b.wav", "lang": "ru", "text": "Мир"}]
        ),
    ]
)

test_params_list.extend(
    [
        # Simple arithmetic expression
        (
            LambdaExpression,
            {"new_field": "duration_x2", "expression": "entry.duration * 2"},
            {"duration": 3.5},
            [{"duration": 3.5, "duration_x2": 7.0}],
        ),

        # Ternary expression
        (
            LambdaExpression,
            {"new_field": "label", "expression": "'long' if entry.duration > 10 else 'short'"},
            {"duration": 12.0},
            [{"duration": 12.0, "label": "long"}],
        ),

        # Filtering: entry should be dropped (condition is False)
        (
            LambdaExpression,
            {"new_field": "valid", "expression": "entry.duration > 10", "filter": True},
            {"duration": 5.0},
            [],
        ),

        # Filtering: entry should be kept (condition is True)
        (
            LambdaExpression,
            {"new_field": "valid", "expression": "entry.duration > 10", "filter": True},
            {"duration": 12.0},
            [{"duration": 12.0, "valid": True}],
        ),

        # Using built-in function len()
        (
            LambdaExpression,
            {"new_field": "num_chars", "expression": "len(entry.text)"},
            {"text": "hello world"},
            [{"text": "hello world", "num_chars": 11}],
        ),

        # Using built-in max() with sub-expressions
        (
            LambdaExpression,
            {"new_field": "score", "expression": "max(entry.a, entry.b * 2)"},
            {"a": 4, "b": 3},
            [{"a": 4, "b": 3, "score": 6}],
        ),

        # Expression using variable prefix (e.g., entry.a + entry.b)
        (
            LambdaExpression,
            {
                "new_field": "sum",
                "expression": "entry.a + entry.b",
                "lambda_param_name": "entry",
            },
            {"a": 1, "b": 2},
            [{"a": 1, "b": 2, "sum": 3}],
        ),

        # Logical expression using `and`
        (
            LambdaExpression,
            {
                "new_field": "check",
                "expression": "entry.a > 0 and entry.b < 5",
            },
            {"a": 1, "b": 4},
            [{"a": 1, "b": 4, "check": True}],
        ),

        # Boolean expression without filtering (entry is always returned)
        (
            LambdaExpression,
            {
                "new_field": "is_zero",
                "expression": "entry.value == 0",
            },
            {"value": 5},
            [{"value": 5, "is_zero": False}],
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

@pytest.mark.parametrize(
    "text,expected_flags",
    [
        # repeated n-grams
        ("yes yes yes yes yes", {"hall_repeated_ngrams": True, "hall_long_word": False, "hall_frequent_single_word": False}),
        # long word
        ("short reallyreallyreallyreallyreallyreallyreallylong", {"hall_repeated_ngrams": False, "hall_long_word": True, "hall_frequent_single_word": False}),
        # known hallucinated phrase
        ("lorem ipsum dolor sit amet", {"hall_repeated_ngrams": False, "hall_long_word": False, "hall_frequent_single_word": True}),
        # no hallucination
        ("this is a normal sentence", {"hall_repeated_ngrams": False, "hall_long_word": False, "hall_frequent_single_word": False}),
    ]
)
def test_detect_whisper_hallucinations(tmp_path, text, expected_flags):
    # prepare common phrases file
    common_phrases_path = tmp_path / "common_phrases.txt"
    common_phrases_path.write_text("lorem ipsum dolor sit amet\n")

    processor = DetectWhisperHallucinationFeatures(
        common_hall_file=str(common_phrases_path),
        output_manifest_file=None  # assuming it's optional or handled elsewhere
    )

    input_entry = {"text": text}
    result_entry = processor.process_dataset_entry(input_entry)[0].data

    # check each expected flag
    for key, value in expected_flags.items():
        assert result_entry[key] == value, f"Failed for text='{text}' on key='{key}'"

@pytest.mark.parametrize("test_class,class_kwargs,test_input,expected_output", test_params_list, ids=str)
def test_data_to_data(test_class, class_kwargs, test_input, expected_output):
    processor = test_class(**class_kwargs, output_manifest_file=None)
    result = [entry.data for entry in processor.process_dataset_entry(test_input)]

    assert result == expected_output