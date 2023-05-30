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
from typing import Dict, List

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.processors.modify_manifest.modify_manifest import ModifyManifestTextProcessor


class MakeLettersUppercaseAfterPeriod(ModifyManifestTextProcessor):
    """
    """

    def __init__(
        self, **kwargs,
    ):
        super().__init__(**kwargs)

    def _process_dataset_entry(self, data_entry) -> List:
        replace_word_counter = collections.defaultdict(int)

        for p in ".!?":
            for letter in "abcdefghijklmnopqrstuvwxyzáéíóúü":
                replace_in = f"{p} {letter}"
                replace_out = f"{p} {letter}".upper()
                if replace_in in data_entry[self.text_key]:
                    data_entry[self.text_key] = data_entry[self.text_key].replace(replace_in, replace_out)
                    replace_word_counter[replace_in] += 1

        return [DataEntry(data=data_entry, metrics=replace_word_counter)]

    def finalize(self, metrics):
        total_counter = collections.defaultdict(int)
        for counter in metrics:
            for word, count in counter.items():
                total_counter[word] += count
        logger.info("Some of the substrings that were substituted")
        total_counter_sorted = dict(sorted(total_counter.items(), key=lambda x: x[1], reverse=True))
        for word, count in total_counter_sorted.items():
            if count > 1:
                logger.info(f"{word} {count}")
        super().finalize(metrics)
