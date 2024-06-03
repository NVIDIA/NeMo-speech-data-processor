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
from typing import List

import toloka.client
import toloka.client.project.template_builder
from docx import Document

from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor


class CreateTolokaTaskSet(BaseProcessor):
    def __init__(
        self,
        input_data_file: str,
        input_pool_file: str,
        limit: float = 100,
        API_KEY: str = "---",
        platform: str = "---",
        pool_id: str = "---",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_data_file = input_data_file
        self.input_pool_file = input_pool_file
        self.limit = limit
        self.API_KEY = API_KEY
        self.platform = platform
        self.pool_id = pool_id

    def prepare(self):
        try:
            with open(self.input_data_file, 'r') as file:
                data = json.loads(file.readline())
                if self.API_KEY == "---":
                    self.API_KEY = data["API_KEY"]
                if self.platform == "---":
                    self.platform = data["platform"]
        except FileNotFoundError:
            print("Data file not found.")

        try:
            with open(self.input_pool_file, 'r') as file:
                data = json.loads(file.readline())
                if self.pool_id == "---":
                    self.pool_id = data["pool_id"]
        except FileNotFoundError:
            print("Pool file not found.")

        self.toloka_client = toloka.client.TolokaClient(self.API_KEY, self.platform)

    def read_manifest(self):
        with open(self.input_manifest_file, "rt") as fin:
            total_lines = sum(1 for line in fin)

            lines_to_read = max(1, int(total_lines * (self.limit / 100)))

            fin.seek(0)

            # Read the specified percentage of lines
            entries = []
            for _ in range(lines_to_read):
                line = fin.readline()
                entries.append(json.loads(line))

        return entries

    def process(self):
        self.prepare()

        entries = self.read_manifest()
        tasks = [
            toloka.client.Task(input_values={'text': data_entry["text"]}, pool_id=self.pool_id)
            for data_entry in entries
        ]

        self.toloka_client.create_tasks(tasks, allow_defaults=True)

        with open(self.output_manifest_file, "wt", encoding='utf-8') as fout:
            for entry in entries:
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
