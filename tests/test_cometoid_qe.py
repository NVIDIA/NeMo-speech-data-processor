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

import pytest
from unittest.mock import MagicMock, patch

from sdp.processors.inference.quality_estimation.pymarian import CometoidWMTQualityEstimation

@pytest.fixture(scope="module")
def mock_processor():
    processor = CometoidWMTQualityEstimation(
        source_text_field="src",
        target_text_field="tgt",
        model_name_or_path="cometoid-wmt23",
        output_field="cometoid_score",
        device_type="cpu",
        num_devices=1,
        chunksize=1,
        output_manifest_file="/tmp/test_output.jsonl",
    )
    return processor


@patch("huggingface_hub.hf_hub_download", return_value="/tmp/dummy")
@patch("sdp.processors.inference.quality_estimation.pymarian.os.path.exists", return_value=True)
@patch("pymarian.Evaluator")
def test_load_model_with_mock(mock_eval, mock_exists, mock_hf_download, mock_processor):
    mock_eval.return_value = MagicMock()
    mock_processor.load_model()
    assert mock_processor.model is not None
    mock_hf_download.assert_called()
    mock_eval.assert_called()


def test_process_dataset_entry(mock_processor):
    mock_processor.model = MagicMock()
    mock_processor.model.evaluate = MagicMock(return_value=[0.875])

    entry = {
        "src": "This is a test sentence.",
        "tgt": "Dies ist ein Testsatz."
    }

    mock_processor._chunk_manifest = lambda: [[entry]]
    mock_processor.finalize = MagicMock()
    mock_processor.number_of_entries = 0

    # Patch load_model to avoid real downloading
    with patch.object(mock_processor, "load_model"), \
         patch("builtins.open"), \
         patch("json.dump"), \
         patch("os.makedirs"):
        mock_processor.process()

    mock_processor.model.evaluate.assert_called_once()
    assert mock_processor.number_of_entries == 1


@pytest.mark.parametrize("source,target", [
    ("Hello", "Hallo"),
    ("Good morning", "Guten Morgen"),
    ("How are you?", "Wie geht's dir?"),
])
def test_score_format(mock_processor, source, target):
    mock_processor.model = MagicMock()
    mock_processor.model.evaluate = MagicMock(return_value=[0.9])

    entry = {"src": source, "tgt": target}
    mock_processor.output_field = "cometoid_score"

    bitext_pairs = [f"{source}\t{target}"]
    scores = mock_processor.model.evaluate(bitext_pairs)

    assert isinstance(scores, list)
    assert len(scores) == 1
    score = scores[0]
    assert 0.0 <= score <= 1.0