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
        super().__init__(**kwargs)
        self.API_KEY = API_KEY or os.getenv('TOLOKA_API_KEY')
        self.platform = platform or os.getenv('TOLOKA_PLATFORM')
        self.project_name = project_name
        self.project_description = project_description
        self.project_instructions = project_instructions
        self.save_api_key_to_config = save_api_key_to_config  # Initialize the parameter
        self.load_config()

    def load_config(self):
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
