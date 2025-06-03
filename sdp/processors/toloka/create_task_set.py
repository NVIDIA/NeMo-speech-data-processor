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

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


try:
    import toloka.client
    import toloka.client.project.template_builder
    TOLOKA_AVAILABLE = True
except ImportError:
    TOLOKA_AVAILABLE = False
    toloka = None




class CreateTolokaTaskSet(BaseParallelProcessor):
    """Creates a set of tasks in a Toloka pool based on user-provided configurations and input data.

    This class reads data from a manifest file, loads the target pool configuration,
    and uses Toloka's API to create and upload tasks into the specified pool.

    Args:
        input_data_file (str): Path to the input data file containing API configurations.
        input_pool_file (str): Path to the input pool file containing pool configurations.
        limit (float): Percentage of tasks to load from the manifest file. Default: 100.

    Returns:
        A set of tasks created and uploaded to the specified Toloka pool.
    """
    def __init__(
        self,
        input_data_file: str,
        input_pool_file: str,
        limit: float = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_data_file = input_data_file
        self.input_pool_file = input_pool_file
        self.limit = limit
        self.pool_id = None
        self.toloka_available = TOLOKA_AVAILABLE
        
        

        
        # Get API key and platform from environment variables
        self.API_KEY = os.getenv('TOLOKA_API_KEY')
        if not self.API_KEY:
            raise ValueError("TOLOKA_API_KEY environment variable is not set")
            
        self.platform = os.getenv('TOLOKA_PLATFORM')
        if not self.platform:
            raise ValueError("TOLOKA_PLATFORM environment variable is not set")
        
        self.toloka_client = None

    def prepare(self):
        """
        Prepares the class by loading pool configuration and initializing Toloka client.

        This method sets up the necessary components for task creation, including loading the
        pool configuration and initializing the Toloka client.
        """
        if self.toloka_available != True:
            logger.warning("Toloka is currently not supported. CreateTaskSet processor functionality will be limited.")
        self.load_pool_config()
        self.toloka_client = toloka.client.TolokaClient(self.API_KEY, self.platform)

    def load_pool_config(self):
        """
        Loads pool configuration data from the input pool file.

        This method reads the pool configuration from the specified file and extracts the
        pool ID for use in task creation.

        Raises:
        ------
        ValueError
            If the input pool file does not contain a pool ID.
        """
        try:
            with open(self.input_pool_file, 'r') as file:
                pool_config = json.load(file)
                self.pool_id = pool_config.get('pool_id')
                if not self.pool_id:
                    raise ValueError("No pool ID found in the pool configuration file.")
        except FileNotFoundError:
            raise ValueError(f"Pool configuration file {self.input_pool_file} not found.")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from the pool configuration file {self.input_pool_file}.")

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

        This method reads the input manifest, creates task objects for Toloka, and submits
        them to the specified pool. It also writes the manifest data to an output file after 
        tasks have been created.

        Raises:
        ------
        ValueError
            If no pool ID is available or if there are issues with the Toloka API.
        """
        logger.info("Processing tasks...")
        self.prepare()

        if not self.pool_id:
            raise ValueError("No pool ID available. Cannot create tasks.")

        entries = self.read_manifest()
        tasks = [
            toloka.client.Task(input_values={'text': data_entry["text"]}, pool_id=self.pool_id)
            for data_entry in entries
        ]

        try:
            self.toloka_client.create_tasks(tasks, allow_defaults=True)
            logger.info(f"Created {len(tasks)} tasks.")
        except Exception as e:
            logger.error(f"Error creating tasks: {e}")
            raise ValueError(f"Failed to create tasks: {e}")

        # Write the manifest data to the output file
        with open(self.output_manifest_file, "wt", encoding='utf-8') as fout:
            for entry in entries:
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
