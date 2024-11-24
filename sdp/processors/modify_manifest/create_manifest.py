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

import pandas as pd

from sdp.processors.base_processor import (
    BaseParallelProcessor,
    BaseProcessor,
    DataEntry,
)


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
        output_file = [str(self.raw_data_dir / file) for file in self.raw_data_dir.rglob('*.' + self.extension)]
        return output_file

    def process_dataset_entry(self, data_entry):
        data = {self.output_file_key: data_entry}
        return [DataEntry(data=data)]


class ReadCsv(BaseProcessor):
    """
    Processor for reading a CSV file and converting its content into a JSON lines format.

    This class reads a CSV file using pandas, processes each row, and writes the output to a specified manifest file in JSON lines format.

    Args:
        header (int, optional): Row number to use as the column names. Defaults to None, meaning no header.
        sep (str, optional): Delimiter to use for separating values in the CSV file. Defaults to ','.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseProcessor`.

    Methods:
        process(): Reads the input CSV file and writes its content as JSON lines to the output manifest file.
    """

    def __init__(
        self,
        header: int = None,
        sep: str = ",",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.header = header
        self.sep = sep

    def process(self):
        df1 = pd.read_csv(self.input_manifest_file, header=self.header, sep=self.sep)
        with open(self.output_manifest_file, "w") as out_file:
            for j, row in df1.iterrows():
                out_file.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
