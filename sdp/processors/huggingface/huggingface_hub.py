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
import os
from typing import Dict

from tqdm.contrib.concurrent import process_map

from sdp.processors.base_processor import BaseProcessor, BaseParallelProcessor

def _hf_hub_download(kwargs):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(**kwargs)

class ListRepoFiles(BaseProcessor):
    """
    Processor that lists files in a Hugging Face Hub repository and writes them
    into a JSON manifest file.

    Each line in the output manifest is a JSON object with the key defined by `file_key`
    and value being a file path from the repository.

    Args:
        output_manifest_file (str): Path to write the output manifest file.
        file_key (str): The key name to use in each manifest entry (default: "file_key").
        **list_repo_files_kwargs: Keyword arguments forwarded to `huggingface_hub.list_repo_files()`.

    See also:
        https://huggingface.co/docs/huggingface_hub/v0.8.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_repo_files

    Returns:
        A line-delimited JSON manifest where each line looks like:
        ``{"file_key": "path/to/file.ext"}``
    """

    def __init__(
        self,
        output_manifest_file: str,
        file_key: str = "file_key",
        **list_repo_files_kwargs,
    ):
        super().__init__(output_manifest_file=output_manifest_file)
        self.list_repo_files_kwargs = list_repo_files_kwargs
        self.file_key = file_key

    def list_repo_files(self):
        """
        Retrieve the list of files from a Hugging Face repository.
        """
        from huggingface_hub import list_repo_files

        self.files = list_repo_files(**self.list_repo_files_kwargs)

    def write_output_manifest_file(self):
        """
        Write the list of repo files to the output manifest, one file per line as JSON.
        """
        with open(self.output_manifest_file, 'w', encoding='utf8') as fout:
            for file in self.files:
                line = json.dumps({self.file_key: file})
                fout.writelines(f'{line}\n')  # Fixed typo: was `lines`, should be `line`

    def process(self):
        """
        Main processing entrypoint: get repo files and write to manifest.
        """
        self.list_repo_files()
        self.write_output_manifest_file()


class SnapshotDownload(BaseProcessor):
    """
    Processor that downloads a snapshot of a Hugging Face repository to a local directory
    and writes the local folder path to a JSON manifest file.

    Args:
        output_manifest_file (str): Path to write the output manifest file.
        input_manifest_file (str, optional): Path to input manifest (not used in this processor).
        **snapshot_download_kwargs: Keyword arguments forwarded to `huggingface_hub.snapshot_download()`.

    See also:
        https://huggingface.co/docs/huggingface_hub/v0.30.2/en/package_reference/file_download#huggingface_hub.snapshot_download

    Returns:
        A JSON file containing one line:
        ``{"destination_dir": "/path/to/downloaded/repo"}``
    """

    def __init__(
        self,
        output_filepath_field: str = "downloaded",
        snapshot_download_args: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_filepath_field = output_filepath_field
        self.snapshot_download_args = snapshot_download_args

    def process(self):
        """
        Main processing entrypoint: download repo and write path to manifest. 
        """
        from huggingface_hub import snapshot_download

        self.local_dir = snapshot_download(**self.snapshot_download_args)
        
        with open(self.output_manifest_file, 'w', encoding='utf8') as fout:
            fout.writelines(json.dumps({self.output_filepath_field : self.local_dir}))


class HfHubDownload(BaseParallelProcessor):
    def __init__(
        self,
        filename_field: str,
        output_filepath_field: str = "downloaded",
        hf_hub_download_args: Dict = {},
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filename_field = filename_field
        self.output_filepath_field = output_filepath_field
        self.hf_hub_download_args = hf_hub_download_args

    def process(self):
        self.prepare()
        os.makedirs(os.path.dirname(self.output_manifest_file), exist_ok=True)

        with open(self.output_manifest_file, "wt", encoding="utf8") as fout:
            for manifest_chunk in self._chunk_manifest():
                # Подготовим список задач
                download_tasks = [
                    {
                        **self.hf_hub_download_args,
                        "filename": entry[self.filename_field]
                    }
                    for entry in manifest_chunk
                ]

                # Параллельная загрузка с учётом max_workers и chunksize
                results = process_map(
                    _hf_hub_download,
                    download_tasks,
                    max_workers=self.max_workers,
                    chunksize=self.chunksize,
                )

                # Сопоставим обратно результаты с входными entry
                for entry, local_path in zip(manifest_chunk, results):
                    entry[self.output_filepath_field] = local_path
                    json.dump(entry, fout, ensure_ascii=False)
                    fout.write("\n")
                    self.number_of_entries += 1

        self.finalize(self.test_cases)
    
    def process_dataset_entry(self, data_entry):
        pass