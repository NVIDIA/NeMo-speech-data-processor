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

import json
import os

from docx import Document

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class CreateSentenceSet(BaseParallelProcessor):
    """Creates a set of sentences from a DOCX file by splitting its content into individual sentences.

    This processor reads a DOCX file, extracts the full text, splits it into sentences
    based on the Armenian period character, and wraps each sentence into a `DataEntry`.

    Args:
        **kwargs: Additional arguments passed to the base `BaseParallelProcessor` class.

    Returns:
        A list of `DataEntry` objects, each containing a single extracted sentence.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse_docx(self, file_path):
        doc = Document(file_path)

        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)

        combined_text = '\n'.join(full_text)

        sentences = combined_text.split('։')

        return sentences

    def process_dataset_entry(self, data_entry):
        file = data_entry["source_filepath"]

        data = [DataEntry(data={"text": text}) for text in self.parse_docx(file)]

        return data
