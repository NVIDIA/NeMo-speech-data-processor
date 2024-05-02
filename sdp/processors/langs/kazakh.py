# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


class LatinToCyrillic(BaseParallelProcessor):
    """Converts visually identical latin letters  to cyrillic equivalents.

    Args:
        text_key (str): a string indicating which key of the data entries
            should be used to find the utterance transcript. Defaults to "text".

    Returns:
         The same data as in the input manifest with latin letters replaced with cyrillic ones.
    """

    LATIN = "AaƏəBEeKkMHOoPpCcTYyXxhi"
    CYRILLIC = "АаӘәВЕеКкМНОоРрСсТУуХхһі"

    def __init__(
        self,
        text_key: str = "text",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_key = text_key

    def process_dataset_entry(self, data_entry) -> List:
        drop_this_utt = False
        latin_counter = collections.defaultdict(int)

        text_in = data_entry[self.text_key]
        text_out = text_in

        for char in text_in:
            if char in self.LATIN:
                cyrillic_eqv = self.CYRILLIC[self.LATIN.index(char)]
                text_out = text_out.replace(char, cyrillic_eqv)
                latin_counter[char] += 1

        data_entry[self.text_key] = text_out
        return [DataEntry(data=data_entry, metrics=latin_counter)]

    def finalize(self, metrics):
        total_counter = collections.defaultdict(int)
        for counter in metrics:
            for char, value in counter.items():
                total_counter[char] += value
        logger.info("Num of Latin characters")
        for char, count in total_counter.items():
            logger.info(f"{char}: {count}")
        super().finalize(metrics)
