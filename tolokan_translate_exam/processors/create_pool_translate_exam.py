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
import json

import toloka.client
import toloka.client.project.template_builder

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class CreateTolokaPoolTranslationExam(BaseParallelProcessor):
    def __init__(self, input_pool_file: str, **kwargs):
        super().__init__(**kwargs)
        self.input_pool_file = input_pool_file

    def prepare(self):
        try:
            with open(self.input_pool_file, 'r') as file:
                data = json.loads(file.readline())
                self.ntraining_pool_id = data["pool_id"]
        except FileNotFoundError:
            print("Pool file not found.")

        return super().prepare()

    def process_dataset_entry(self, data_entry):
        API_KEY = data_entry["API_KEY"]
        project_id = data_entry["project_id"]
        platform = data_entry["platform"]
        toloka_client = toloka.client.TolokaClient(API_KEY, platform)

        translator_skill = next(toloka_client.get_skills(name='Russian-Armenian translation'), None)
        if translator_skill:
            print(f'{translator_skill.name} skill exists. ID: {translator_skill.id}')
        else:
            translator_skill = toloka_client.create_skill(
                name='Russian-Armenian translation',
                public_requester_description={
                    'EN': 'Translation detection skill from russian to armenian',
                    'HY': 'Ռուսերենից հայերեն թարգմանության դետեկցայի որակը',
                    'RU': 'Качество oпределения правильности перевода с русского на армянский',
                },
            )
            print(f'Skill created. ID: {translator_skill.id}')

        new_pool = toloka.client.Pool(
            project_id=project_id,
            private_name='Переводы экзамен',
            may_contain_adult_content=False,
            will_expire=datetime.datetime.utcnow() + datetime.timedelta(days=365),
            reward_per_assignment=0.05,
            assignment_max_duration_seconds=60 * 10,
            filter=((toloka.client.filter.Languages.in_('HY')) & (toloka.client.filter.Languages.in_('RU'))),
            defaults=toloka.client.pool.Pool.Defaults(default_overlap_for_new_task_suites=1),
        )

        print(self.ntraining_pool_id)
        new_pool.quality_control.training_requirement = (
            toloka.client.quality_control.QualityControl.TrainingRequirement(
                training_pool_id=self.ntraining_pool_id, training_passing_skill_value=100
            )
        )
        # training_requirement=toloka.client.quality_control.QualityControl.TrainingRequirement(training_pool_id=self.training_pool_id),

        new_pool.quality_control.add_action(
            collector=toloka.client.collectors.GoldenSet(),
            conditions=[
                toloka.client.conditions.GoldenSetCorrectAnswersRate > 80,
            ],
            action=toloka.client.actions.SetSkillFromOutputField(
                skill_id=translator_skill.id,
                from_field='correct_answers_rate',
            ),
        )

        new_pool.set_mixer_config(golden_tasks_count=10)
        new_pool = toloka_client.create_pool(new_pool)

        data = {"pool_id": new_pool.id}

        return [DataEntry(data=data)]
