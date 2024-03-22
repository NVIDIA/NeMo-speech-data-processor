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
import toloka.client.project.template_builder

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class GetTolokaResults(BaseParallelProcessor):
    def __init__(
        self,
        input_data_file: str,
        input_pool_file: str,
        output_dir: str,
        status: str = "ACCEPTED",
        API_KEY: str = "---",
        platform: str = "---",
        pool_id: str = "---",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_data_file = input_data_file
        self.input_pool_file = input_pool_file
        self.output_dir = output_dir
        self.status = status
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
