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
import os

import toloka.client
import toloka.client.project.template_builder

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class CreateTolokaPool(BaseParallelProcessor):
    """
    CreateTolokaPool is a class for creating pools on the Toloka crowdsourcing platform.
    This class uses Toloka's API to create pools for a given project, based on user-provided configurations.

    Attributes:
    ----------
    API_KEY : str, optional
        The API key used to authenticate with Toloka's API. Defaults to None, in which case it tries to
        load the key from environment variables or config file.
    platform : str, optional
        Specifies the Toloka environment (e.g., 'PRODUCTION', 'SANDBOX'). Defaults to None, meaning it will
        try to load from environment variables or the config file.
    project_id : str, optional
        The ID of the project for which the pool will be created. This can be provided during initialization
        or loaded from the configuration file.
    lang : str, optional
        The language filter for the pool. Defaults to 'HY'.

    Methods:
    -------
    load_config()
        Loads configuration data from a manifest file to populate API_KEY, platform, and project_id attributes.
    process_dataset_entry(data_entry)
        Creates a new Toloka pool based on the provided dataset entry.
    setup_quality_control(pool)
        Sets up quality control rules for the pool to ensure data quality.
    """
    def __init__(
        self,
        API_KEY: str = None,
        platform: str = None,
        project_id: str = None,  # Optional project_id during initialization
        lang: str = 'HY',
        **kwargs,
    ):
        """
        Constructs the necessary attributes for the CreateTolokaPool class.

        Parameters:
        ----------
        API_KEY : str, optional
            The API key used to authenticate with Toloka's API. If not provided, it is retrieved from the environment.
        platform : str, optional
            Specifies the Toloka environment (e.g., 'PRODUCTION', 'SANDBOX'). If not provided, it is retrieved from the environment.
        project_id : str, optional
            The ID of the project for which the pool will be created. Defaults to None.
        lang : str, optional
            The language filter for the pool. Defaults to 'HY'.
        """
        super().__init__(**kwargs)
        self.API_KEY = API_KEY or os.getenv('TOLOKA_API_KEY')
        self.platform = platform or os.getenv('TOLOKA_PLATFORM')
        self.project_id = project_id  # Store project_id if provided during initialization
        self.lang = lang
        self.load_config()

    def load_config(self):
        """
        Loads configuration data from the output manifest file to populate API_KEY, platform, and project_id attributes.

        This method attempts to read from a JSON configuration file. If the file is missing or improperly
        formatted, an appropriate error is logged.
        """
        try:
            with open(self.output_manifest_file, 'r') as file:
                config = json.load(file)
                self.API_KEY = config.get('API_KEY', self.API_KEY)
                self.platform = config.get('platform', self.platform)
                self.project_id = config.get('project_id', self.project_id)
        except FileNotFoundError:
            logger.error("Configuration file not found.")
        except json.JSONDecodeError:
            logger.error("Error decoding JSON from the configuration file.")

    def process_dataset_entry(self, data_entry):
        """
        Creates a new Toloka pool based on the provided dataset entry.

        This method retrieves necessary information such as API key, project ID, and platform from the
        dataset entry or defaults to the instance's attributes. It then uses Toloka's API to create a
        new pool for the specified project and returns the pool details.

        Parameters:
        ----------
        data_entry : dict
            A dictionary containing the data entry information, which may include overrides for API_KEY,
            project_id, or platform.

        Returns:
        -------
        list
            A list containing a DataEntry object with the new pool ID if successful, or an empty list if failed.
        """
        API_KEY = data_entry.get("API_KEY", self.API_KEY)  # Retrieve API_KEY from data_entry or use self.API_KEY
        project_id = data_entry.get(
            "project_id", self.project_id
        )  # Retrieve project_id from data_entry or use self.project_id
        platform = data_entry.get("platform", self.platform)

        try:
            toloka_client = toloka.client.TolokaClient(API_KEY, platform)

            new_pool = toloka.client.Pool(
                project_id=project_id,
                private_name='Voice recording',
                may_contain_adult_content=False,
                will_expire=datetime.datetime.utcnow() + datetime.timedelta(days=365),
                reward_per_assignment=0.01,
                assignment_max_duration_seconds=60 * 10,
                auto_accept_solutions=False,
                auto_accept_period_day=14,
                filter=(
                    (toloka.client.filter.Languages.in_(self.lang)) & (toloka.client.filter.ClientType == 'TOLOKA_APP')
                ),
            )
            new_pool.set_mixer_config(real_tasks_count=5)
            self.setup_quality_control(new_pool)

            new_pool = toloka_client.create_pool(new_pool)
            data = {"pool_id": new_pool.id}
            return [DataEntry(data=data)]
        except Exception as e:
            logger.error(f"Failed to create a new pool in Toloka: {e}")
            return []

    def setup_quality_control(self, pool):
        """
        Sets up quality control rules for the Toloka pool to ensure high-quality task results.

        Parameters:
        ----------
        pool : toloka.client.Pool
            The pool object for which quality control rules will be set up.
        """
        # Control for skipped tasks in a row
        pool.quality_control.add_action(
            collector=toloka.client.collectors.SkippedInRowAssignments(),
            conditions=[toloka.client.conditions.SkippedInRowCount >= 2],
            action=toloka.client.actions.RestrictionV2(
                scope='POOL',
                duration=1,
                duration_unit='DAYS',
                private_comment='Skips too many task suites in a row',
            ),
        )

        # Control for fast responses that might indicate fraud
        pool.quality_control.add_action(
            collector=toloka.client.collectors.AssignmentSubmitTime(history_size=10, fast_submit_threshold_seconds=60),
            conditions=[toloka.client.conditions.FastSubmittedCount >= 5],
            action=toloka.client.actions.RestrictionV2(
                scope='ALL_PROJECTS',
                duration_unit='PERMANENT',
                private_comment='Fast responses',
            ),
        )
