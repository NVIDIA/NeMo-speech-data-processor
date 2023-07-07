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

import collections
import os
import re
import tempfile
import urllib.request
from typing import List

import pandas as pd

from sdp.logging import logger
from sdp.processors.base_processor import DataEntry
from sdp.processors.modify_manifest.modify_manifest import ModifyManifestTextProcessor


class CleanRomanNumerals(ModifyManifestTextProcessor):
    def __init__(
        self,
        king_triggers,
        queen_triggers,
        ordinal_masc_triggers,
        ordinal_fem_triggers,
        cardinal_triggers,
        numerals_data_path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.numerals_data_path = numerals_data_path
        self.king_triggers = king_triggers
        self.queen_triggers = queen_triggers
        self.ordinal_masc_triggers = ordinal_masc_triggers
        self.ordinal_fem_triggers = ordinal_fem_triggers
        self.cardinal_triggers = cardinal_triggers

        # read csv
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "1-100_roman_numeral_table.csv")
        df = pd.read_csv(csv_path, sep="\t", index_col=0)

        self.roman_numeral_to_ordinal_masc = {}
        for i, row in df.iterrows():
            self.roman_numeral_to_ordinal_masc[row["roman"]] = row["ordinal_masc"].strip()

        self.roman_numeral_to_ordinal_fem = {}
        for i, row in df.iterrows():
            self.roman_numeral_to_ordinal_fem[row["roman"]] = row["ordinal_fem"].strip()

        self.roman_numeral_to_cardinal = {}
        for i, row in df.iterrows():
            self.roman_numeral_to_cardinal[row["roman"]] = row["cardinal"].strip()

        self.roman_numeral_to_king = {}
        for i, row in df.iterrows():
            self.roman_numeral_to_king[row["roman"]] = row["king"].strip()

        self.roman_numeral_to_queen = {}
        for i, row in df.iterrows():
            self.roman_numeral_to_queen[row["roman"]] = row["queen"].strip()

        self.clean_roman_numerals_count = collections.defaultdict(int)

    def _process_dataset_entry(self, data_entry) -> List:
        data_entry = self.clean_operation(data_entry, self.ordinal_masc_triggers, self.roman_numeral_to_ordinal_masc)
        data_entry = self.clean_operation(data_entry, self.ordinal_fem_triggers, self.roman_numeral_to_ordinal_fem)
        data_entry = self.clean_operation(data_entry, self.cardinal_triggers, self.roman_numeral_to_cardinal)
        data_entry = self.clean_operation(data_entry, self.king_triggers, self.roman_numeral_to_king)
        data_entry = self.clean_operation(data_entry, self.queen_triggers, self.roman_numeral_to_queen)
        return [DataEntry(data=data_entry, metrics=self.clean_roman_numerals_count)]

    def finalize(self, metrics):
        total_counter = collections.defaultdict(int)
        for counter in metrics:
            for word, count in counter.items():
                total_counter[word] += count
        logger.info("Num of roman numeral substitutions")
        total_counter_sorted = dict(
            sorted(
                total_counter.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )
        for word, count in total_counter_sorted.items():
            logger.info(f"{word} {count}")
        super().finalize(metrics)

    def clean_operation(self, data, triggers, roman_numeral_to_num_written):
        for trigger in triggers:
            trigger_match = re.search(
                pattern=f"({trigger} \S*)\s",
                string=data[self.text_key],
                flags=re.IGNORECASE,
            )

            if trigger_match:
                trigger_numeral = trigger_match.group(0).strip()
                trigger, numeral = trigger_numeral.split(" ")

                if numeral.lower() in roman_numeral_to_num_written:
                    number = roman_numeral_to_num_written[numeral.lower()]

                    if trigger[0].isupper():
                        # 'felipe iv' --> 'felipe cuarto'
                        # 'Felipe iv' --> 'Felipe Cuarto'
                        number = number.capitalize()

                    trigger_number = f"{trigger} {number}"

                    data[self.text_key] = data[self.text_key].replace(trigger_numeral, trigger_number)
                    self.clean_roman_numerals_count[trigger_numeral] += 1
        return data
