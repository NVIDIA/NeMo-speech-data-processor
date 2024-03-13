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
from collections import defaultdict

import toloka.client
import toloka.client.project.template_builder
from docx import Document

from sdp.processors.base_processor import (
    BaseParallelProcessor,
    BaseProcessor,
    DataEntry,
)

# class AcceptIfWERLess(BaseParallelProcessor):
#     def __init__(
#         self,
#         input_data_file: str,
#         input_pool_file: str,
#         trashhold: float = 75,
#         API_KEY: str = "---",
#         platform: str = "---",
#         pool_id: str = "---",
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         self.input_data_file = input_data_file
#         self.input_pool_file = input_pool_file
#         self.trashhold = trashhold
#         self.API_KEY = API_KEY
#         self.platform = platform
#         self.pool_id = pool_id

#     def prepare(self):
#         try:
#             with open(self.input_data_file, 'r') as file:
#                 data = json.loads(file.readline())
#                 if self.API_KEY == "---":
#                     self.API_KEY = data["API_KEY"]
#                 if self.platform == "---":
#                     self.platform = data["platform"]
#         except FileNotFoundError:
#             print("Data file not found.")

#         try:
#             with open(self.input_pool_file, 'r') as file:
#                 data = json.loads(file.readline())
#                 if self.pool_id == "---":
#                     self.pool_id = data["pool_id"]
#         except FileNotFoundError:
#             print("Pool file not found.")

#         self.toloka_client = toloka.client.TolokaClient(self.API_KEY, self.platform)

#         return super().prepare()

#     def process_dataset_entry(self, data_entry):
#         if data_entry["wer"] < self.trashhold:
#             if str(data_entry["status"]) == "Status.SUBMITTED":
#                 self.toloka_client.accept_assignment(assignment_id=data_entry["task_id"], public_comment='Well done!')
#             return [DataEntry(data=None)]

#         return [DataEntry(data=data_entry)]


class AcceptIfWERLess(BaseProcessor):
    def __init__(
        self,
        input_data_file: str,
        input_pool_file: str,
        threshold: float = 75,
        API_KEY: str = "---",
        platform: str = "---",
        pool_id: str = "---",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_data_file = input_data_file
        self.input_pool_file = input_pool_file
        self.threshold = threshold
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

    def process(self):
        big_dict = defaultdict(int)
        self.prepare()
        with open(self.input_manifest_file, 'r') as file:
            for line in file:
                data_entry = json.loads(line)
                if data_entry["wer"] < self.threshold:
                    if str(data_entry["status"]) == "Status.SUBMITTED":
                        big_dict[data_entry["assignment_id"]] += 1

        for assignment_id, count in big_dict.items():
            if count >= 3:  # should be >= 3 or == 5
                self.toloka_client.accept_assignment(assignment_id=assignment_id, public_comment='Well done!')
