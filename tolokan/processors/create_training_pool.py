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

import toloka.client
import toloka.client.project.template_builder

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class CreateTolokaPool(BaseParallelProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process_dataset_entry(self, data_entry):
        API_KEY = data_entry["API_KEY"]
        project_id = data_entry["project_id"]
        platform = data_entry["platform"]
        toloka_client = toloka.client.TolokaClient(API_KEY, platform)

        training_pool = toloka.training.Training(
            project_id=project_id,
            private_name='Training pool',
            training_tasks_in_task_suite_count=5,
            task_suites_required_to_pass=1,
            may_contain_adult_content=False,
            inherited_instructions=True,
            assignment_max_duration_seconds=60 * 5,
            retry_training_after_days=5,
            mix_tasks_in_creation_order=True,
            shuffle_tasks_in_task_suite=True,
        )
        training_pool = toloka_client.create_training(training_pool)

        data = {"pool_id": training_pool.id}

        return [DataEntry(data=data)]
