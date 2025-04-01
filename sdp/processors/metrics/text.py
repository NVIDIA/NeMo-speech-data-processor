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

import re

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry

class CountNumWords(BaseParallelProcessor):
    """
    Processor for counting the number of words in the text_key field saving the number in num_words_key.

    Args:
        text_key (str): The field containing the input text in the dataset.
        num_words_key (str): The field to store the number of words in the dataset.
        alphabet (str): Characters to be used to count words. Any other characters are substituted by whitespace and not take into account.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    """

    def __init__(
        self,
        text_key: str,
        num_words_key: str,
        alphabet: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_key = text_key
        self.num_words_key = num_words_key
        self.pattern = re.compile("[^" + alphabet + "]")

    def process_dataset_entry(self, data_entry):
        text = data_entry[self.text_key]
        cleaned_string = self.pattern.sub("", text).strip()
        cleaned_string = re.sub("\\s+", " ", cleaned_string).strip()
        words = cleaned_string.split()
        num_words = len(words)
        data_entry[self.num_words_key] = num_words
        return [DataEntry(data=data_entry)]