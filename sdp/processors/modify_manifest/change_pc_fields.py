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

from typing import Dict

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry

NA_INDICATOR = "n/a"


class ChangePCFields(BaseParallelProcessor):
    """Getting text from either text_pc or text_pc_pred and marking the origin
    with either "original" or "synthetic".
    """
    def __init__(
        self, **kwargs,
    ):
        super().__init__(**kwargs)

    def process_dataset_entry(self, data_entry: Dict):
        if data_entry["text_pc"] != NA_INDICATOR:
            data_entry["text"] = data_entry["text_pc"]
            data_entry["text_pc_origin"] = "original"

        else:
            data_entry["text"] = data_entry["text_pc_pred"]
            data_entry["text_pc_origin"] = "synthetic"

        # remove old fields
        del data_entry["text_pc"]
        del data_entry["text_pc_pred"]

        return [DataEntry(data=data_entry)]
