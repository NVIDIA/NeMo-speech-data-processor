# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from huggingface_hub import snapshot_download, list_repo_files

from sdp.processors.base_processor import BaseProcessor


class ListRepoFiles(BaseProcessor):
    def __init__(
        self,
        output_manifest_file,
        input_manifest_file: str = None,
        **list_repo_files_kwargs,
    ):
        super().__init__(output_manifest_file = output_manifest_file, input_manifest_file = input_manifest_file)
        self.list_repo_files_kwargs = list_repo_files_kwargs

    def list_repo_files(self):
        self.files = list_repo_files(**self.list_repo_files_kwargs)
    
    def write_output_manifest_file(self):
        with open(self.output_manifest_file, 'w', encoding='utf8') as fout: 
            for file in self.files:
                line = json.dumps(dict(file_key = file))
                fout.writelines(f'{lines}\n')
    
    def process(self):
        self.list_repo_files()
        self.write_output_manifest_file()


class SnapshotDownload(BaseProcessor):
    def __init__(
        self,
        output_manifest_file,
        input_manifest_file: str = None,
        **snapshot_download_kwargs,
    ):
        super().__init__(output_manifest_file = output_manifest_file, input_manifest_file = input_manifest_file)
        self.snapshot_download_kwargs = snapshot_download_kwargs

    def download(self):
        self.local_dir = snapshot_download(**self.snapshot_download_kwargs)
    
    def write_output_manifest_file(self):
        with open(self.output_manifest_file, 'w', encoding='utf8') as fout:
            fout.writelines(json.dumps(dict(destination_dir = self.local_dir)))

    def process(self):
        self.download()
        self.write_output_manifest_file()