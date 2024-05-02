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


class CreateTolokaPool(BaseParallelProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process_dataset_entry(self, data_entry):
        API_KEY = data_entry["API_KEY"]
        project_id = data_entry["project_id"]
        platform = data_entry["platform"]
        toloka_client = toloka.client.TolokaClient(API_KEY, platform)

        new_pool = toloka.client.Pool(
            project_id=project_id,
            private_name='Recording Validation',
            may_contain_adult_content=False,
            will_expire=datetime.datetime.utcnow() + datetime.timedelta(days=365),
            reward_per_assignment=0.01,
            defaults=toloka.client.Pool.Defaults(default_overlap_for_new_task_suites=3),
            assignment_max_duration_seconds=60 * 10,
            auto_accept_solutions=False,
            auto_accept_period_day=14,
            filter=((toloka.client.filter.Languages.in_('HY'))),
        )
        new_pool.set_mixer_config(real_tasks_count=9, golden_tasks_count=1)
        # Setting up quality control for skipped assignments
        new_pool.quality_control.add_action(
            collector=toloka.client.collectors.SkippedInRowAssignments(),
            conditions=[toloka.client.conditions.SkippedInRowCount >= 2],
            action=toloka.client.actions.RestrictionV2(
                scope='POOL',
                duration=1,
                duration_unit='DAYS',
                private_comment='Skips too many task suites in a row',
            ),
        )
        # Fast responses fraud
        new_pool.quality_control.add_action(
            collector=toloka.client.collectors.AssignmentSubmitTime(history_size=10, fast_submit_threshold_seconds=60),
            conditions=[toloka.client.conditions.FastSubmittedCount >= 5],
            action=toloka.client.actions.RestrictionV2(
                scope='ALL_PROJECTS',
                duration_unit='PERMANENT',
                private_comment='Fast responses',
            ),
        )
        # Majority vote
        new_pool.quality_control.add_action(
            collector=toloka.client.collectors.MajorityVote(answer_threshold=2),
            conditions=[
                toloka.client.conditions.TotalAnswersCount > 9,
                toloka.client.conditions.CorrectAnswersRate < 60,
            ],
            action=toloka.client.actions.RejectAllAssignments(
                public_comment='Too low quality'
            ),  # Myabe we should ban?
        )

        new_pool = toloka_client.create_pool(new_pool)

        data = {"pool_id": new_pool.id}

        return [DataEntry(data=data)]
