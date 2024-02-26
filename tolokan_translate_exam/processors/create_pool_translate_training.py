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

import datetime

import toloka.client
import toloka.client.project.template_builder

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class CreateTolokaPoolTranslationTraining(BaseParallelProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process_dataset_entry(self, data_entry):
        API_KEY = data_entry["API_KEY"]
        project_id = data_entry["project_id"]
        platform = data_entry["platform"]
        toloka_client = toloka.client.TolokaClient(API_KEY, platform)

        new_training = toloka.client.training.Training(
            project_id=project_id,
            private_name='Тренировка перед прохождением экзамена. ',
            may_contain_adult_content=False,
            assignment_max_duration_seconds=60 * 10,
            mix_tasks_in_creation_order=True,
            shuffle_tasks_in_task_suite=True,
            training_tasks_in_task_suite_count=10,
            task_suites_required_to_pass=1,
            retry_training_after_days=1,
            inherited_instructions=True,
            public_instructions='Это часть с публичными инструкциями которая не была унаследована.',
        )
        new_training = toloka_client.create_training(new_training)

        data = {"pool_id": new_training.id}

        return [DataEntry(data=data)]
