# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from sdp.processors.inference.nlp.fasttext.fasttext import FastTextLangIdClassifier


@pytest.fixture(scope="module")
def classifier():
    processor = FastTextLangIdClassifier(
        model_name_or_path="lid.176.ftz",
        text_field="text",
        output_field="lang",
        num_workers=1,
        batch_size=1,
    )
    processor.prepare()
    return processor


@pytest.mark.parametrize("text,expected_lang", [
    ("Hello, how are you?", "en"),
    ("Bonjour tout le monde", "fr"),
    ("Привет, как дела?", "ru"),
    ("Hola, ¿cómo estás?", "es"),
])
def test_language_identification(classifier, text, expected_lang):
    input_entry = {"text": text}
    result = classifier.process_dataset_entry(input_entry)

    assert isinstance(result, list)
    assert len(result) == 1

    output = result[0].data
    assert "lang" in output
    assert "lang_prob" in output

    predicted_lang = output["lang"]
    prob = output["lang_prob"]

    assert isinstance(predicted_lang, str)
    assert 0 <= prob <= 1.0

    #Exact matching may depend on the model, so we compare based on presence in the top predictions.
    assert predicted_lang == expected_lang, f"Expected: {expected_lang}, got: {predicted_lang}"