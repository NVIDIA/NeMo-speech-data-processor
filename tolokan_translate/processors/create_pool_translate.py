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


class CreateTolokaPoolTranslate(BaseParallelProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process_dataset_entry(self, data_entry):
        API_KEY = data_entry["API_KEY"]
        project_id = data_entry["project_id"]
        platform = data_entry["platform"]
        toloka_client = toloka.client.TolokaClient(API_KEY, platform)

        new_pool = toloka.client.Pool(
            project_id=project_id,
            private_name='Основные переводы',
            may_contain_adult_content=False,
            will_expire=datetime.datetime.utcnow() + datetime.timedelta(days=365),
            reward_per_assignment=0.01,
            assignment_max_duration_seconds=60 * 10,
            auto_accept_solutions=False,
            auto_accept_period_day=15,
            filter=(
                (toloka.client.filter.Languages.in_('HY'))
                & (toloka.client.filter.Languages.in_('RU'))
                & (toloka.client.filter.ClientType == 'TOLOKA_APP')
            ),
        )
        new_pool.set_mixer_config(real_tasks_count=5)
        new_pool = toloka_client.create_pool(new_pool)

        data = {"pool_id": new_pool.id}

        return [DataEntry(data=data)]
