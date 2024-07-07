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
import logging
import os
import sys

import toloka.client
import toloka.client.project.template_builder

from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor

# class CreateTolokaProject(BaseProcessor):
#     def __init__(
#         self,
#         API_KEY: str,
#         platform: str,
#         project_name: str,
#         project_description: str,
#         project_instructions: str,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.API_KEY = API_KEY
#         self.platform = platform
#         self.project_name = project_name
#         self.project_description = project_description
#         self.project_instructions = project_instructions

#     def process(self):
#         logging.basicConfig(
#             format='[%(levelname)s] %(name)s: %(message)s',
#             level=logging.INFO,
#             stream=sys.stdout,
#         )

#         toloka_client = toloka.client.TolokaClient(self.API_KEY, self.platform)  # 'PRODUCTION' or 'SANDBOX'

#         new_project = toloka.client.Project(
#             public_name=self.project_name,
#             public_description=self.project_description,
#             public_instructions=self.project_instructions,
#         )

#         text_view = toloka.client.project.template_builder.TextViewV1(
#             toloka.client.project.template_builder.InputData('text')
#         )
#         audio_field = toloka.client.project.template_builder.AudioFieldV1(
#             toloka.client.project.template_builder.OutputData('audio_file'),
#             validation=toloka.client.project.template_builder.RequiredConditionV1(),
#         )
#         width_plugin = toloka.client.project.template_builder.TolokaPluginV1('scroll', task_width=500)

#         project_interface = toloka.client.project.TemplateBuilderViewSpec(
#             view=toloka.client.project.template_builder.ListViewV1(items=[text_view, audio_field]),
#             plugins=[width_plugin],
#         )

#         input_specification = {'text': toloka.client.project.StringSpec()}
#         output_specification = {'audio_file': toloka.client.project.FileSpec()}

#         new_project.task_spec = toloka.client.project.task_spec.TaskSpec(
#             input_spec=input_specification,
#             output_spec=output_specification,
#             view_spec=project_interface,
#         )

#         new_project = toloka_client.create_project(new_project)

#         data_file = self.output_manifest_file
#         directory = os.path.dirname(data_file)
#         if not os.path.exists(directory):
#             os.makedirs(directory)

#         data = {"API_KEY": self.API_KEY, "project_id": new_project.id, "platform": self.platform}
#         with open(data_file, "w") as fout:
#             fout.write(json.dumps(data) + "\n")


class CreateTolokaProject(BaseProcessor):
    def __init__(
        self,
        project_name: str,
        project_description: str,
        project_instructions: str,
        config_file: str = None,
        API_KEY: str = None,
        platform: str = None,
        save_to_file: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config_file = config_file
        self.API_KEY = API_KEY or os.getenv('TOLOka_API_KEY')
        self.platform = platform or os.getenv('TOLOka_PLATFORM')
        self.project_name = project_name
        self.project_description = project_description
        self.project_instructions = project_instructions
        self.save_to_file = save_to_file
        if self.config_file:
            self.load_config()

    def load_config(self):
        try:
            with open(self.config_file, 'r') as file:
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

        # Optionally save API key and project details to a file
        if self.save_to_file:
            data_file = self.output_manifest_file
            directory = os.path.dirname(data_file)
            if not os.path.exists(directory):
                os.makedirs(directory)

            data = {"API_KEY": self.API_KEY, "project_id": created_project.id, "platform": self.platform}
            with open(data_file, "w") as fout:
                fout.write(json.dumps(data) + "\n")

        logger.info("Project created successfully: Project ID - {}".format(created_project.id))
