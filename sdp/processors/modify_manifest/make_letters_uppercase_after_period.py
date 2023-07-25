# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import collections
from typing import List

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry

# TODO: should be done with general sub-regex processor


class MakeLettersUppercaseAfterPeriod(BaseParallelProcessor):
    """Can be used to replace characters with upper-case version after punctuation.

    Args:
        punctuation (str): string with all punctuation characters to consider.
            Defaults to ".!?".
        text_key (str): a string indicating which key of the data entries
            should be used to find an utterance transcript. Defaults to "text".

    Returns:
         The same data as in the input manifest with ``<text_key>`` field changed.
    """

    def __init__(
        self, punctuation=".!?", text_key: str = "text", **kwargs,
    ):
        super().__init__(**kwargs)
        self.punctuation = punctuation
        self.text_key = text_key

    def process_dataset_entry(self, data_entry) -> List:
        replace_word_counter = collections.defaultdict(int)

        # keeping in a list, since strings are immutable
        new_text = []

        idx = 0
        while idx < len(data_entry[self.text_key]):
            character = data_entry[self.text_key][idx]
            # checking that next is space and then we upper whatever is after that
            # note that Python's upper correctly does not change anything that's not a letter
            if (
                character in self.punctuation
                and idx + 2 < len(data_entry[self.text_key])
                and data_entry[self.text_key][idx + 1] == " "
            ):
                new_text.extend([character, " ", data_entry[self.text_key][idx + 2].upper()])
                replace_word_counter[data_entry[self.text_key][idx : idx + 3]] += 1
                idx += 2
            else:
                new_text.append(character)
            idx += 1
        data_entry[self.text_key] = "".join(new_text)

        return [DataEntry(data=data_entry, metrics=replace_word_counter)]

    def finalize(self, metrics):
        total_counter = collections.defaultdict(int)
        for counter in metrics:
            for word, count in counter.items():
                total_counter[word] += count
        logger.info("Some of the substrings that were uppercased")
        total_counter_sorted = dict(sorted(total_counter.items(), key=lambda x: x[1], reverse=True))
        for word, count in total_counter_sorted.items():
            if count > 1:
                logger.info(f"{word} {count}")
        super().finalize(metrics)
