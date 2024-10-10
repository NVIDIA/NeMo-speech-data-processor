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
import os
from collections import defaultdict

import toloka.client
import toloka.client.project.template_builder
from tqdm import tqdm

from sdp.processors.base_processor import BaseProcessor


class AcceptIfWERLess(BaseProcessor):
    def __init__(
        self,
        input_data_file: str,
        input_pool_file: str,
        threshold: float = 75,
        config_file: str = None,
        API_KEY: str = None,
        platform: str = None,
        pool_id: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_data_file = input_data_file
        self.input_pool_file = input_pool_file
        self.threshold = threshold
        self.config_file = config_file
        self.API_KEY = API_KEY or os.getenv('TOLOKA_API_KEY')
        self.platform = platform or os.getenv('TOLOKA_PLATFORM')
        self.pool_id = pool_id
        if self.config_file:
            self.load_config()

    def load_config(self):
        try:
            with open(self.config_file, 'r') as file:
                config = json.load(file)
                self.API_KEY = config.get('API_KEY', self.API_KEY)
                self.platform = config.get('platform', self.platform)
                self.pool_id = config.get('pool_id', self.pool_id)
        except FileNotFoundError:
            print("Configuration file not found.")
        except json.JSONDecodeError:
            print("Error decoding JSON from the configuration file.")

    def prepare(self):
        if not self.API_KEY or not self.platform or not self.pool_id:
            try:
                with open(self.input_data_file, 'r') as file:
                    data = json.loads(file.readline())
                    self.API_KEY = data.get("API_KEY", self.API_KEY)
                    self.platform = data.get("platform", self.platform)
            except FileNotFoundError:
                print("Data file not found.")
            except json.JSONDecodeError:
                print("Error decoding JSON from the data file.")

            try:
                with open(self.input_pool_file, 'r') as file:
                    data = json.loads(file.readline())
                    self.pool_id = data.get("pool_id", self.pool_id)
            except FileNotFoundError:
                print("Pool file not found.")
            except json.JSONDecodeError:
                print("Error decoding JSON from the pool file.")

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

        accepted = 0
        for assignment_id, count in tqdm(big_dict.items()):
            if count >= 3:  # should be >= 3 and <= 5
                self.toloka_client.accept_assignment(assignment_id=assignment_id, public_comment='Well done!')
                accepted += 1

        print(f"Number of accepted task suits: {accepted} of {len(big_dict)}")
