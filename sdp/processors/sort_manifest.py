# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from sdp.processors.base_processor import BaseProcessor


class SortManifest(BaseProcessor):
    """
    Processor which will sort the manifest by some specified attribute.

    Args:
        output_manifest: the path to the output manifest. It will be the same as the 
            input manifest, but resorted.
        input_manifest_file: the path to the input manifest which will be resorted.
        attribute_sort_by: the attribute by which the manifest will be sorted.
        descending: if set to False (default), attribute will be in ascending order. 
            If True, attribute will be in descending order.
    
    """

    def __init__(
        self, output_manifest_file: str, input_manifest_file: str, attribute_sort_by: str, descending: bool = True
    ):
        self.output_manifest_file = output_manifest_file
        self.input_manifest_file = input_manifest_file
        self.attribute_sort_by = attribute_sort_by
        self.descending = descending

    def process(self):

        with open(self.input_manifest_file, "rt", encoding="utf8") as fin:
            dataset_entries = [json.loads(line) for line in fin.readlines()]

        dataset_entries = sorted(dataset_entries, key=lambda x: x[self.attribute_sort_by], reverse=self.descending)

        with open(self.output_manifest_file, "wt", encoding="utf8") as fout:
            for line in dataset_entries:
                fout.write(json.dumps(line) + "\n")
