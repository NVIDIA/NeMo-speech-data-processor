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

import toloka.client

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class GetTolokaResults(BaseParallelProcessor):
    def __init__(
        self,
        input_data_file: str,
        input_pool_file: str,
        output_dir: str,
        status: str = "ACCEPTED",
        config_file: str = None,
        API_KEY: str = None,
        platform: str = None,
        pool_id: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_data_file = input_data_file
        self.input_pool_file = input_pool_file
        self.output_dir = output_dir
        self.status = status
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
            logger.error("Configuration file not found.")
        except json.JSONDecodeError:
            logger.error("Error decoding JSON from the configuration file.")

    def prepare(self):
        if not self.API_KEY or not self.platform or not self.pool_id:
            try:
                with open(self.input_data_file, 'r') as file:
                    data = json.loads(file.readline())
                    self.API_KEY = data.get("API_KEY", self.API_KEY)
                    self.platform = data.get("platform", self.platform)
            except FileNotFoundError:
                logger.error("Data file not found.")
            except json.JSONDecodeError:
                logger.error("Error decoding JSON from the data file.")

            try:
                with open(self.input_pool_file, 'r') as file:
                    data = json.loads(file.readline())
                    self.pool_id = data.get("pool_id", self.pool_id)
            except FileNotFoundError:
                logger.error("Pool file not found.")
            except json.JSONDecodeError:
                logger.error("Error decoding JSON from the pool file.")

        self.toloka_client = toloka.client.TolokaClient(self.API_KEY, self.platform)
        return super().prepare()

    def read_manifest(self):
        for assignment in self.toloka_client.get_assignments(pool_id=self.pool_id):
            if str(assignment.status) == 'Status.' + self.status:
                # ACCEPTED — Accepted by the requester.
                # ACTIVE — Being picked up by a Toloker.
                # EXPIRED — The time for completing the tasks expired.
                # REJECTED — Rejected by the requester.
                # SKIPPED — Skipped by the Toloker.
                # SUBMITTED — Completed but not checked.
                if (
                    str(assignment.status) == 'Status.ACCEPTED'
                    or str(assignment.status) == 'Status.REJECTED'
                    or str(assignment.status) == 'Status.SUBMITTED'
                ):
                    for task, solution in zip(assignment.tasks, assignment.solutions):
                        suit_id = assignment.task_suite_id
                        assignment_id = assignment.id
                        user_id = assignment.user_id
                        task_id = task.id
                        text = task.input_values['text']
                        attachment_id = solution.output_values.get('audio_file', None)
                        status = assignment.status
                        task_info = {
                            'task_id': task_id,
                            'text': text,
                            'attachment_id': attachment_id,
                            'status': str(status),
                            'suit_id': suit_id,
                            'assignment_id': assignment_id,
                            'user_id': user_id,
                        }
                        yield task_info
                else:
                    for task in assignment.tasks:
                        suit_id = assignment.task_suite_id
                        assignment_id = assignment.id
                        user_id = assignment.user_id
                        task_id = task.id
                        text = task.input_values['text']
                        attachment_id = ""
                        status = assignment.status
                        task_info = {
                            'task_id': task_id,
                            'text': text,
                            'attachment_id': attachment_id,
                            'status': str(status),
                            'suit_id': suit_id,
                            'assignment_id': assignment_id,
                            'user_id': user_id,
                        }
                        yield task_info

    def process_dataset_entry(self, data_entry):
        user_id = data_entry["user_id"]
        task_id = data_entry["task_id"]
        text = data_entry["text"]
        attachment_id = data_entry["attachment_id"]
        status = data_entry["status"]
        suit_id = data_entry["suit_id"]
        assignment_id = data_entry["assignment_id"]
        output_path = os.path.join(self.output_dir, attachment_id + '.wav')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if attachment_id != "":
            with open(output_path, 'wb') as attachment_file:
                self.toloka_client.download_attachment(attachment_id, out=attachment_file)

        task_info = {
            'task_id': task_id,
            'text': text,
            'attachment_id': attachment_id,
            'status': status,
            'audio_filepath': output_path,
            'suit_id': suit_id,
            'assignment_id': assignment_id,
            'user_id': user_id,
        }

        return [DataEntry(data=task_info)]
