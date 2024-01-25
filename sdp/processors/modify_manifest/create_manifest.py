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

from pathlib import Path

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class CreateInitialManifestByExt(BaseParallelProcessor):
    """
    Processor for creating an initial dataset manifest by saving filepaths with a common extension to the field specified in output_field.

    Args:
        raw_data_dir (str): The root directory of the files to be added to the initial manifest. This processor will recursively look for files with the extension 'extension' inside this directory.
        output_field (str): The field to store the paths to the files in the dataset.
        extension (str): The field stecify extension of the files to use them in the dataset.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    """

    def __init__(
        self,
        raw_data_dir: str,
        output_field: str = "audio_filepath",
        extension: str = "mp3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        self.output_field = output_field
        self.extension = extension

    def read_manifest(self):
        input_files = [str(self.raw_data_dir / file) for file in self.raw_data_dir.rglob('*.' + self.extension)]
        return input_files

    def process_dataset_entry(self, data_entry):
        data = {self.output_field: data_entry}
        return [DataEntry(data=data)]
