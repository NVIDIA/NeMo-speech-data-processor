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

import pandas

from sdp.processors.base_processor import BaseProcessor


class ExcelToJsonConverter(BaseProcessor):
    def __init__(self, input_excel_file: str, input_souce: str, output_source: str, **kwargs):
        super().__init__(**kwargs)
        self.input_excel_file = input_excel_file
        self.input_souce = input_souce
        self.output_source = output_source

    def process(self):
        df = pandas.read_excel(self.input_excel_file, header=0)

        data_entries = []

        for _, row in df.iterrows():
            input_source = row[0]
            output_source = row[1]

            data_entries.append({self.input_source: input_source, self.output_source: output_source})

        with open(self.output_json_file, "wt", encoding='utf-8') as fout:
            json.dump(data_entries, fout, ensure_ascii=False, indent=4)
