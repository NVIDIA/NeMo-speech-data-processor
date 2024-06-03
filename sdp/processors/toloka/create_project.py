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

from sdp.processors.base_processor import BaseProcessor


class CreateTolokaProject(BaseProcessor):
    def __init__(
        self,
        API_KEY: str,
        platform: str,
        project_name: str,
        project_description: str,
        project_instructions: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.API_KEY = API_KEY
        self.platform = platform
        self.project_name = project_name
        self.project_description = project_description
        self.project_instructions = project_instructions

    def process(self):
        logging.basicConfig(
            format='[%(levelname)s] %(name)s: %(message)s',
            level=logging.INFO,
            stream=sys.stdout,
        )

        toloka_client = toloka.client.TolokaClient(self.API_KEY, self.platform)  # 'PRODUCTION' or 'SANDBOX'

        new_project = toloka.client.Project(
            public_name=self.project_name,
            public_description=self.project_description,
            public_instructions=self.project_instructions,
        )

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

        input_specification = {'text': toloka.client.project.StringSpec()}
        output_specification = {'audio_file': toloka.client.project.FileSpec()}

        new_project.task_spec = toloka.client.project.task_spec.TaskSpec(
            input_spec=input_specification,
            output_spec=output_specification,
            view_spec=project_interface,
        )

        new_project = toloka_client.create_project(new_project)

        data_file = self.output_manifest_file
        directory = os.path.dirname(data_file)
        if not os.path.exists(directory):
            os.makedirs(directory)

        data = {"API_KEY": self.API_KEY, "project_id": new_project.id, "platform": self.platform}
        with open(data_file, "w") as fout:
            fout.write(json.dumps(data) + "\n")
