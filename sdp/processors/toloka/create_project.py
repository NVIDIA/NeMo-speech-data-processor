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

import toloka.client
import toloka.client.project.template_builder

from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor


class CreateTolokaProject(BaseProcessor):
    """
    CreateTolokaProject is a class for creating projects on the Toloka crowdsourcing platform.
    This class leverages Toloka's API to create a project based on user-provided configurations.

    Attributes:
    ----------
    project_name : str
        The name of the project to be created.
    project_description : str
        A description of the project that will be shown to the Toloka workers.
    project_instructions : str
        Instructions that will be provided to the workers on how to complete tasks for the project.
    API_KEY : str, optional
        The API key used to authenticate with Toloka's API. Defaults to None, in which case it tries to
        load the key from environment variables or config file.
    platform : str, optional
        Specifies the Toloka environment (e.g., 'PRODUCTION', 'SANDBOX'). Defaults to None, meaning it will
        try to load from environment variables or the config file.
    save_api_key_to_config : bool, optional
        If True, saves the API key to the configuration file for future use. Defaults to False.

    Methods:
    -------
    load_config()
        Loads configuration data from a manifest file to populate API_KEY and platform attributes.
    process()
        Creates a Toloka project based on provided configurations, saving project details to a file.
    """
    def __init__(
        self,
        project_name: str,
        project_description: str,
        project_instructions: str,
        API_KEY: str = None,
        platform: str = None,
        save_api_key_to_config: bool = False,  # Parameter to control saving API key
        **kwargs,
    ):
        """
        Constructs the necessary attributes for the CreateTolokaProject class.

        Parameters:
        ----------
        project_name : str
            The name of the project to be created.
        project_description : str
            A description of the project that will be shown to the Toloka workers.
        project_instructions : str
            Instructions that will be provided to the workers on how to complete tasks for the project.
        API_KEY : str, optional
            The API key used to authenticate with Toloka's API. If not provided, it is retrieved from the environment.
        platform : str, optional
            Specifies the Toloka environment (e.g., 'PRODUCTION', 'SANDBOX'). If not provided, it is retrieved from the environment.
        save_api_key_to_config : bool, optional
            Determines if the API key should be saved to the configuration file. Defaults to False.
        """
        super().__init__(**kwargs)
        self.API_KEY = API_KEY or os.getenv('TOLOKA_API_KEY')
        self.platform = platform or os.getenv('TOLOKA_PLATFORM')
        self.project_name = project_name
        self.project_description = project_description
        self.project_instructions = project_instructions
        self.save_api_key_to_config = save_api_key_to_config  # Initialize the parameter
        self.load_config()

    def load_config(self):
        """
        Loads configuration data from the output manifest file to populate API_KEY and platform attributes.

        This method attempts to read from a JSON configuration file. If the file is missing or improperly
        formatted, an appropriate error is logged.
        """
        try:
            with open(self.output_manifest_file, 'r') as file:
                config = json.load(file)
                self.API_KEY = config.get('API_KEY', self.API_KEY)
                self.platform = config.get('platform', self.platform)
        except FileNotFoundError:
            logger.error("Configuration file not found.")
        except json.JSONDecodeError:
            logger.error("Error decoding JSON from the configuration file.")

    def process(self):
        """
        Processes the creation of a Toloka project.

        This method establishes a connection to the Toloka API using the provided API key and platform,
        then creates a new project with the specified name, description, and instructions. It also defines
        the task specifications including the input and output fields, and then submits the project to Toloka.

        After creating the project, it saves the project details (including the project ID) to a specified file.
        Optionally, the API key can also be saved to the configuration file.
        """
        logger.info("Processing Toloka project creation...")

        toloka_client = toloka.client.TolokaClient(self.API_KEY, self.platform)

        # Create a new project
        new_project = toloka.client.Project(
            public_name=self.project_name,
            public_description=self.project_description,
            public_instructions=self.project_instructions,
        )

        # Setup the project interface
        text_view = toloka.client.project.template_builder.TextViewV1(
            toloka.client.project.template_builder.InputData('text')
        )
        audio_field = toloka.client.project.template_builder.AudioFieldV1(
            toloka.client.project.template_builder.OutputData('audio_file'),
            validation=toloka.client.project.template_builder.RequiredConditionV1(),
        )
        width_plugin = toloka.client.project.template_builder.TolokaPluginV1('scroll', task_width=500)

        project_interface = toloka.client.project.TemplateBuilderViewSpec(
            view=toloka.client.project.template_builder.ListViewV1(items=[text_view, audio_field]),
            plugins=[width_plugin],
        )

        # Define task specification
        input_specification = {'text': toloka.client.project.StringSpec()}
        output_specification = {'audio_file': toloka.client.project.FileSpec()}

        new_project.task_spec = toloka.client.project.task_spec.TaskSpec(
            input_spec=input_specification,
            output_spec=output_specification,
            view_spec=project_interface,
        )

        # Create the project in Toloka
        created_project = toloka_client.create_project(new_project)

        # Always save project details and possibly the API key to a file
        data_file = self.output_manifest_file
        directory = os.path.dirname(data_file)
        if not os.path.exists(directory):
            os.makedirs(directory)

        data = {"project_id": created_project.id, "platform": self.platform}
        if self.save_api_key_to_config:
            data["API_KEY"] = self.API_KEY

        with open(data_file, "w") as fout:
            fout.write(json.dumps(data) + "\n")

        logger.info("Project created successfully: Project ID - {}".format(created_project.id))

