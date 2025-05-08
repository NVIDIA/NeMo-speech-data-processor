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
from collections import defaultdict
from typing import Optional

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry

try:
    import toloka.client
    import toloka.client.project.template_builder
    TOLOKA_AVAILABLE = True
except ImportError:
    TOLOKA_AVAILABLE = False
    toloka = None
    

from tqdm import tqdm


class AcceptIfWERLess(BaseParallelProcessor):
    """This processor accepts Toloka assignments if the Word Error Rate (WER) is below a threshold.

    It evaluates the WER between ground truth and predicted text for each assignment
    and accepts those that meet the specified threshold criteria.

    Args:
        input_data_file (str): Path to the input data file containing API configurations.
        input_pool_file (str): Path to the input pool file containing pool configurations.
        threshold (float): The WER threshold below which assignments are accepted. Default: 75.
        config_file (str): Path to the configuration file. Default: None.
        API_KEY (str): The API key for authenticating with Toloka's API. Default: None.
        platform (str): The Toloka platform to use. Default: None.
        pool_id (str): The ID of the Toloka pool. Default: None.

    Returns:
        A manifest with accepted assignments from Toloka based on the WER threshold.
        
    Example:
    .. code-block:: yaml

        - _target_: sdp.processors.toloka.accept_if.AcceptIfWERLess
            input_manifest_file: ${workspace_dir}/result_manifest_pred_clean.json
            output_manifest_file: ${workspace_dir}/result_manifest_pred_review.json
            input_data_file: ${workspace_dir}/data_file.json
            input_pool_file: ${workspace_dir}/taskpool.json
            threshold: 50
    """
    
    def __init__(
        self,
        input_data_file: str,
        input_pool_file: str,
        threshold: float = 75,
        config_file: str = None,
        API_KEY: str = None,
        platform: str = None,
        pool_id: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_data_file = input_data_file
        self.input_pool_file = input_pool_file
        self.threshold = threshold
        self.config_file = config_file
        self.API_KEY = API_KEY or os.getenv('TOLOKA_API_KEY')
        self.platform = platform or os.getenv('TOLOKA_PLATFORM')
        self.pool_id = pool_id
        if self.config_file:
            self.load_config()
        self.toloka_available = TOLOKA_AVAILABLE

    def load_config(self):
        """
        Loads configuration data from the specified config file.

        This method attempts to read configuration details such as API key, platform, and pool ID from a JSON file.
        If the file is missing or improperly formatted, an appropriate error is logged.
        """
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
        """
        Prepares the class by loading API configuration, pool configuration, and initializing Toloka client.

        This method loads necessary configurations and initializes the Toloka client to interact with Toloka's API.
        """
        if self.toloka_available != True:
            logger.warning("Toloka is currently not supported. AcceptIf processor functionality will be limited.")

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

    def process(self):
        """
        Accepts Toloka assignments if their Word Error Rate (WER) is below the specified threshold.

        This method reads assignments from the manifest file, evaluates the WER, and accepts assignments that
        meet the acceptance criteria.
        """
        big_dict = defaultdict(int)
        self.prepare()
        with open(self.input_manifest_file, 'r') as file:
            for line in file:
                data_entry = json.loads(line)
                if data_entry["wer"] < self.threshold:
                    if str(data_entry["status"]) == "Status.SUBMITTED":
                        big_dict[data_entry["assignment_id"]] += 1

        accepted = 0
        for assignment_id, count in tqdm(big_dict.items()):
            if count >= 3:  # should be >= 3 and <= 5
                self.toloka_client.accept_assignment(assignment_id=assignment_id, public_comment='Well done!')
                accepted += 1

        logger.info(f"Number of accepted task suits: {accepted} of {len(big_dict)}")

