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

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry

try:
    import toloka.client
    TOLOKA_AVAILABLE = True
except ImportError:
    TOLOKA_AVAILABLE = False
    toloka = None
    



class GetTolokaResults(BaseParallelProcessor):
    """Fetches and stores results from a specified Toloka pool based on user-configured conditions.

    This class connects to Toloka, retrieves task results from a specified pool, filters them by assignment status,
    and stores the results in the given output directory.

    Args:
        input_data_file (str): Path to the input data file containing API configurations.
        input_pool_file (str): Path to the input pool file containing pool configurations.
        output_dir (str): Directory where the results will be stored.
        status (str): Status filter for assignments to retrieve (default: 'ACCEPTED').
        config_file (str): Path to a configuration file. Default: None.
        API_KEY (str): The API key for authenticating with Toloka's API. Default: None.
        platform (str): The Toloka environment to use ('PRODUCTION' or 'SANDBOX'). Default: None.
        pool_id (str): The ID of the Toloka pool to retrieve results from. Default: None.

    Returns:
        A set of task results from Toloka, stored in the specified output directory.
    """
    def __init__(
        self,
        input_data_file: str,
        input_pool_file: str,
        output_dir: str,
        status: str = "ACCEPTED",
        config_file: str = None,
        API_KEY: str = None,
        platform: str = None,
        pool_id: str = None,
        **kwargs
    ):
        """
        Constructs the necessary attributes for the GetTolokaResults class.

        Parameters:
        ----------
        input_data_file : str
            The path to the input data file containing API configurations.
        input_pool_file : str
            The path to the input pool file containing pool configurations.
        output_dir : str
            The directory where the output results will be stored.
        status : str, optional
            The status filter for assignments to retrieve. Defaults to 'ACCEPTED'.
        config_file : str, optional
            The path to the configuration file. Defaults to None.
        API_KEY : str, optional
            The API key used to authenticate with Toloka's API. If not provided, it is retrieved from the environment.
        platform : str, optional
            Specifies the Toloka environment (e.g., 'PRODUCTION', 'SANDBOX'). If not provided, it is retrieved from the environment.
        pool_id : str, optional
            The ID of the pool from which results will be retrieved. Defaults to None.
        """
        super().__init__(**kwargs)
        self.input_data_file = input_data_file
        self.input_pool_file = input_pool_file
        self.output_dir = output_dir
        self.status = status
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
            logger.warning("Toloka is currently not supported. DownloadResponses processor functionality will be limited.")

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
        return super().prepare()

    def read_manifest(self):
        """
        Retrieves and yields task information from Toloka based on the specified pool and assignment status.

        This method retrieves assignments from Toloka for a given pool and yields task information for
        each assignment that matches the specified status.

        Yields:
        ------
        dict
            A dictionary containing task information such as task ID, text, attachment ID, status, etc.
        """
        for assignment in self.toloka_client.get_assignments(pool_id=self.pool_id):
            if str(assignment.status) == 'Status.' + self.status:
                # ACCEPTED, ACTIVE, EXPIRED, REJECTED, SKIPPED, SUBMITTED
                if (
                    str(assignment.status) == 'Status.ACCEPTED'
                    or str(assignment.status) == 'Status.REJECTED'
                    or str(assignment.status) == 'Status.SUBMITTED'
                ):
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
                else:
                    for task in assignment.tasks:
                        suit_id = assignment.task_suite_id
                        assignment_id = assignment.id
                        user_id = assignment.user_id
                        task_id = task.id
                        text = task.input_values['text']
                        attachment_id = ""
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
        """
        Downloads and processes individual task results.

        This method takes a data entry, retrieves the corresponding attachment, and stores it in the
        specified output directory. The task information is then returned.

        Parameters:
        ----------
        data_entry : dict
            A dictionary containing the data entry information.

        Returns:
        -------
        list
            A list containing a DataEntry object with the task information.
        """
        user_id = data_entry["user_id"]
        task_id = data_entry["task_id"]
        text = data_entry["text"]
        attachment_id = data_entry["attachment_id"]
        status = data_entry["status"]
        suit_id = data_entry["suit_id"]
        assignment_id = data_entry["assignment_id"]
        output_path = os.path.join(self.output_dir, attachment_id + '.wav')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if attachment_id != "":
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

