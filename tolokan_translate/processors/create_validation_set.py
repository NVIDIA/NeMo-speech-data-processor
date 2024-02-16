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

import xlsxwriter
from docx import Document

from sdp.processors.base_processor import BaseProcessor


class CreateValidationSet(BaseProcessor):
    def __init__(self, limit: float = 100, lines_to_read: int = -1, output_excel_file: str = "output.xlsx", **kwargs):
        super().__init__(**kwargs)
        self.limit = limit
        self.lines_to_read = lines_to_read
        self.output_excel_file = output_excel_file
        self.workbook = xlsxwriter.Workbook(self.output_excel_file)
        self.worksheet = self.workbook.add_worksheet()
        self.row = 0

    def prepare(self):
        self.worksheet.write(self.row, 0, 'Input Text')
        self.worksheet.write(self.row, 1, 'Output Text')
        self.row += 1

    def read_manifest(self, input_manifest_file):
        data_entries = []
        with open(input_manifest_file, "rt") as fin:
            total_lines = sum(1 for line in fin)
            lines_to_read = max(1, int(total_lines * (self.limit / 100)))
            fin.seek(0)

            if self.lines_to_read != -1:
                lines_to_read = min(lines_to_read, self.lines_to_read)

            for _ in range(lines_to_read):
                line = fin.readline()
                data_entries.append(json.loads(line))
        return data_entries

    def process(self):
        self.prepare()

        data_entries = self.read_manifest(self.input_manifest_file)

        for data_entry in data_entries:
            input_text = data_entry["text"]
            output_text = ""

            self.worksheet.write(self.row, 0, input_text)
            self.worksheet.write(self.row, 1, output_text)
            self.row += 1

        self.workbook.close()
