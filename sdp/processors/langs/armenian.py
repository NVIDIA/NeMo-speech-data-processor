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

import os
from pathlib import Path

import pandas as pd

from sdp.processors.base_processor import (
    BaseParallelProcessor,
    BaseProcessor,
    DataEntry,
)
from sdp.utils.common import load_manifest


class GetSourceBookName(BaseParallelProcessor):
    """
    Processor for extracting source book name from file paths and updating the manifest.

    Args:
        source_file_key (str): The field containing the file path in the manifest.
        source_key (str): The field to store the extracted source book name in the manifest.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    """

    def __init__(
        self,
        source_file_key: str,
        source_key: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.source_file_key = source_file_key
        self.source_key = source_key

    def process_dataset_entry(self, data_entry):
        input_values = os.path.splitext(data_entry[self.source_file_key])[0].split("/")

        data_entry[self.source_key] = input_values[-1]
        return [DataEntry(data=data_entry)]


class MakeTsv(BaseProcessor):
    """
    Processor for converting a JSON manifest file to a TSV (Tab-Separated Values) file.

    Args:
        **kwargs: Additional keyword arguments to be passed to the base class `BaseProcessor`.

    """

    def process(self):
        df1 = pd.DataFrame.from_records(load_manifest(Path(self.input_manifest_file)))
        df1.to_csv(self.output_manifest_file, index=None, sep='\t')


class RandomTsvPart(BaseProcessor):
    """
    Processor for creating a random subset of a TSV (Tab-Separated Values) file based on the specified fraction.

    Args:
        part (float): The fraction of the dataset to include in the random subset, should be in the range (0.0, 1.0).
        random_state (int): Seed for reproducibility when generating the random subset.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseProcessor`.

    """

    def __init__(
        self,
        part: float,
        random_state: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.part = part
        self.random_state = random_state

    def process(self):
        df1 = pd.read_csv(self.input_manifest_file, sep='\t')
        df1.sample(frac=self.part, random_state=self.random_state).to_csv(
            self.output_manifest_file, index=None, sep='\t'
        )
