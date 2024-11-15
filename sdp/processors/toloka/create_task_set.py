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
from typing import List, Optional

import toloka.client
import toloka.client.project.template_builder

from sdp.logging import (
    logger,  # Assuming the logger is properly configured in this module
)
from sdp.processors.base_processor import BaseProcessor


class CreateTolokaTaskSet(BaseProcessor):
    """
    CreateTolokaTaskSet is a class for creating task sets on the Toloka crowdsourcing platform.
    This class uses Toloka's API to create task sets based on user-provided configurations and input data.

    Attributes:
    ----------
    input_data_file : str
        The path to the input data file containing API configurations.
    input_pool_file : str
        The path to the input pool file containing pool configurations.
    limit : float, optional
        The percentage limit of tasks to read from the manifest file. Defaults to 100.
    API_KEY : str, optional
        The API key used to authenticate with Toloka's API. Defaults to None, in which case it tries to
        load the key from environment variables or config file.
    platform : str, optional
        Specifies the Toloka environment (e.g., 'PRODUCTION', 'SANDBOX'). Defaults to None, meaning it will
        try to load from environment variables or the config file.
    pool_id : str, optional
        The ID of the pool to which tasks will be added. Defaults to None, meaning it should be provided in a config file.

    Methods:
    -------
    prepare()
        Prepares the class by loading API configuration, pool configuration, and initializing Toloka client.
    load_api_config()
        Loads API configuration data from the input data file.
    load_pool_config()
        Loads pool configuration data from the input pool file.
    read_manifest() -> List[dict]
        Reads and returns a portion of the manifest data from the input manifest file based on the specified limit.
    process()
        Creates Toloka tasks based on manifest data and adds them to the specified pool.
    """
    def __init__(
        self,
        input_data_file: str,
        input_pool_file: str,
        limit: float = 100,
        API_KEY: Optional[str] = None,
        platform: Optional[str] = None,
        pool_id: Optional[str] = None,
        **kwargs,
    ):
        """
        Constructs the necessary attributes for the CreateTolokaTaskSet class.

        Parameters:
        ----------
        input_data_file : str
            The path to the input data file containing API configurations.
        input_pool_file : str
            The path to the input pool file containing pool configurations.
        limit : float, optional
            The percentage limit of tasks to read from the manifest file. Defaults to 100.
        API_KEY : str, optional
            The API key used to authenticate with Toloka's API. If not provided, it is retrieved from the environment.
        platform : str, optional
            Specifies the Toloka environment (e.g., 'PRODUCTION', 'SANDBOX'). If not provided, it is retrieved from the environment.
        pool_id : str, optional
            The ID of the pool to which tasks will be added. Defaults to None.
        """
        super().__init__(**kwargs)
        self.input_data_file = input_data_file
        self.input_pool_file = input_pool_file
        self.limit = limit
        self.API_KEY = API_KEY or os.getenv('TOLOKA_API_KEY')
        self.platform = platform or os.getenv('TOLOKA_PLATFORM')
        self.pool_id = pool_id

    def prepare(self):
        """
        Prepares the class by loading API configuration, pool configuration, and initializing Toloka client.

        This method loads necessary configurations and initializes the Toloka client to interact with Toloka's API.
        """
        logger.info("Preparing task set...")
        self.load_api_config()
        self.load_pool_config()
        self.toloka_client = toloka.client.TolokaClient(self.API_KEY, self.platform)

    def load_api_config(self):
        """
        Loads API configuration data from the input data file.

        This method attempts to read API configuration, such as API key and platform, from a JSON file.
        If the file is missing or improperly formatted, an appropriate error is logged.
        """
        try:
            with open(self.input_data_file, 'r') as file:
                data = json.load(file)
                self.API_KEY = data.get("API_KEY", self.API_KEY)
                self.platform = data.get("platform", self.platform)
        except FileNotFoundError:
            logger.error("API config file not found.")
        except json.JSONDecodeError:
            logger.error("Error decoding JSON from the API config file.")

    def load_pool_config(self):
        """
        Loads pool configuration data from the input pool file.

        This method attempts to read pool configuration, such as pool ID, from a JSON file.
        If the file is missing or improperly formatted, an appropriate error is logged.
        """
        try:
            with open(self.input_pool_file, 'r') as file:
                data = json.load(file)
                self.pool_id = data.get("pool_id", self.pool_id)
        except FileNotFoundError:
            logger.error("Pool config file not found.")
        except json.JSONDecodeError:
            logger.error("Error decoding JSON from the pool config file.")

    def read_manifest(self) -> List[dict]:
        """
        Reads and returns a portion of the manifest data from the input manifest file based on the specified limit.

        This method reads the input manifest file, calculates the number of entries to read based on the
        specified limit, and returns a list of those entries.

        Returns:
        -------
        List[dict]
            A list of manifest data entries that have been read.
        """
        logger.info("Reading manifest...")
        with open(self.input_manifest_file, "rt") as fin:
            total_lines = sum(1 for _ in fin)
            lines_to_read = max(1, int(total_lines * (self.limit / 100)))
            fin.seek(0)
            entries = [json.loads(fin.readline()) for _ in range(lines_to_read)]
            return entries

    def process(self):
        """
        Creates Toloka tasks based on manifest data and adds them to the specified pool.

        This method reads the manifest data, creates tasks for each data entry, and adds those tasks to the
        specified Toloka pool. It also writes the manifest data to an output file after tasks have been created.
        """
        logger.info("Processing tasks...")
        self.prepare()

        entries = self.read_manifest()
        tasks = [
            toloka.client.Task(input_values={'text': data_entry["text"]}, pool_id=self.pool_id)
            for data_entry in entries
        ]

        self.toloka_client.create_tasks(tasks, allow_defaults=True)
        logger.info(f"Created {len(tasks)} tasks.")

        with open(self.output_manifest_file, "wt", encoding='utf-8') as fout:
            for entry in entries:
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
