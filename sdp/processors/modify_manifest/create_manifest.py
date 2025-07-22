# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from pathlib import Path

from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import Task

from sdp.processors.base_processor import (
    BaseParallelProcessor,
    BaseProcessor,
    DataEntry,
)


class SaveJsonl(BaseProcessor):
    """
    Processor for saving tasks as a one JSONL file.

    Args:
        **kwargs: Additional keyword arguments to be passed to the base class `BaseProcessor`.

    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def setup_on_node(self, _, __):
        open(self.output_manifest_file, 'w').close()

    def process(self, tasks: DataEntry) -> DataEntry:
        with open(self.output_manifest_file, 'a', encoding="utf8") as f:
            f.write(json.dumps(tasks.data) + '\n')
        return tasks


class CreateInitialManifestByExt(BaseParallelProcessor):
    """
    Processor for creating an initial dataset manifest by saving filepaths with a common extension to the field specified in output_field.

    Args:
        raw_data_dir (str): The root directory of the files to be added to the initial manifest. This processor will recursively look for files with the extension 'extension' inside this directory.
        output_file_key (str): The key to store the paths to the files in the dataset.
        extension (str): The file extension of the of the files to be added to the manifest.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    """

    def __init__(
        self,
        raw_data_dir: str,
        output_file_key: str = "audio_filepath",
        extension: str = "mp3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        self.output_file_key = output_file_key
        self.extension = extension

    def read_manifest(self):
        # Get all files with the specified extension
        files = list(self.raw_data_dir.rglob('*.' + self.extension))
        # Get relative paths and then rebuild proper paths to avoid duplication
        return [str(self.raw_data_dir / file.relative_to(self.raw_data_dir)) for file in files]

    def process_dataset_entry(self, data_entry):
        data = {self.output_file_key: data_entry}
        return [
            DataEntry(
                data=data,
                task_id=0,
                dataset_name=str(self.raw_data_dir / "*.") + self.extension,
            )
        ]


class CreateCombinedManifests(BaseParallelProcessor):
    """Reads JSON lines from specified files and creates a combined manifest.

    This processor iterates over files listed in `manifest_list`, reads each file line by line,
    and yields the parsed JSON data from each line.

    Args:
        manifest_list (list(str)): A list of file paths or directories to process. The processor will
                                   recursively read files within the directories and expect each file to contain JSON data.
        **kwargs: Additional keyword arguments passed to the base class `BaseParallelProcessor`.

    Returns:
        A generator that yields parsed JSON data from each line in the files listed in `manifest_list`.
    """

    def __init__(
        self,
        manifest_list: list[str],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.manifest_list = manifest_list

    def read_manifest(self):
        for file in self.manifest_list:
            with open(file, "rt", encoding="utf8") as fin:
                for line in fin:
                    yield json.loads(line)

    def process_dataset_entry(self, data_entry):
        return [
            DataEntry(
                data=data_entry,
                task_id=0,
                dataset_name=self.__class__.__name__,
            )
        ]
