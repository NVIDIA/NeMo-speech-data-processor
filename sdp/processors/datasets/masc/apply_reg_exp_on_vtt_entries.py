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

import os
import re
import webvtt # pip install webvtt-py
from typing import Dict
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class RegExpVttEntries(BaseParallelProcessor):
    """
    Applies regular expressions on entries of a .vtt (WebVTT) file and stores the processed file in the specified directory.

    Args::
        input_filepath_key (str): Key that stores path to the input `.vtt` file.
        output_filtered_vtt_dir (str): Directory where the processed `.vtt` files will be stored.
        output_filepath_key (str): Key to store the output `.vtt` file path.

    Returns::
        Manifest with additional field:
        {
            "output_filepath_key": <path to processed .vtt file>
        }
    """

    def __init__(
        self,
        regex_params: Dict,
        input_filepath_key: str = "vtt_filepath",
        output_filtered_vtt_dir: str = "filtered_vtt_filepath",
        output_filepath_key: str = "filtered_vtt_filepath",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_filepath_key = input_filepath_key
        self.output_filepath_key = output_filepath_key
        self.output_filtered_vtt_dir = output_filtered_vtt_dir
        self.regex_params = regex_params

    def prepare(self):
        os.makedirs(self.output_filtered_vtt_dir, exist_ok=True)

    def process_dataset_entry(self, data_entry):
        try:
            vtt = webvtt.read(data_entry[self.input_filepath_key])

            for caption in vtt:
                caption.text = re.sub(
                    pattern=self.regex_params["pattern"],
                    repl=self.regex_params["repl"],
                    string=caption.text,
                    count=self.regex_params.get("count", 0),
                )

            basename = os.path.basename(data_entry[self.input_filepath_key])
            filtered_vtt_filepath = os.path.join(self.output_filtered_vtt_dir, basename)
            data_entry[self.output_filepath_key] = filtered_vtt_filepath

            vtt.save(filtered_vtt_filepath)
            return [DataEntry(data=data_entry)]
        except:
            return [DataEntry(data=None)]
