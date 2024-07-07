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

import toloka.client
import toloka.client.project.template_builder
from docx import Document
from tqdm import tqdm

from sdp.processors.base_processor import BaseProcessor


class RejectIfBanned(BaseProcessor):
    def __init__(
        self,
        input_data_file: str,
        input_pool_file: str,
        API_KEY: str = "---",
        platform: str = "---",
        pool_id: str = "---",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_data_file = input_data_file
        self.input_pool_file = input_pool_file
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
        self.prepare()
        list_of_banned = []
        reject_list = []
        list_of_banned = [
            restriction.user_id for restriction in self.toloka_client.get_user_restrictions(scope='ALL_PROJECTS')
        ]
        print("LIST OF BANNED -------------------------", list_of_banned)
        with open(self.input_manifest_file, 'r') as file:
            for line in file:
                data_entry = json.loads(line)
                if data_entry["user_id"] in list_of_banned:
                    if str(data_entry["status"]) == "Status.SUBMITTED":
                        if data_entry['assignment_id'] not in reject_list:
                            reject_list.append(data_entry['assignment_id'])

        print("REJECTION LIST -------------------------", reject_list)
        for assignment_id in tqdm(reject_list, desc="Rejecting assignments"):
            self.toloka_client.reject_assignment(assignment_id=assignment_id, public_comment='Bad quality of audio.')
