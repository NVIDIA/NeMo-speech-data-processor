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

import os
import shutil
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import itertools
from tqdm.contrib.concurrent import process_map


from sdp.logging import logger
from sdp.processors.base_processor import DataEntry, BaseParallelProcessor


class RemoveFiles(BaseParallelProcessor):
    """
    A processor that removes files or directories from the filesystem based on a filepath
    specified in the input data entry.

    This processor is typically used for cleanup tasks after processing files.

    Args:
        filepath_field (str): The key in the data entry that holds the path to the file or directory to remove.
        
        drop_filepath_field (bool): Whether to remove the filepath field from the resulting data entry. Defaults to True.

        recursive (bool): Whether to recursively remove files from directories. Defaults to False.
        
        **kwargs: Additional arguments passed to the BaseParallelProcessor.
    
    Returns:
        A manifest where each entry is the same as the input, optionally without the filepath field,
        and with the file or directory at the specified path removed from disk.
    
    Example entry before processing::
    
        {
            "id": "abc123",
            "path_to_remove": "/tmp/some_file.wav"
        }
    
    Example entry after processing (if `drop_filepath_field=True`)::
    
        {
            "id": "abc123"
        }
    """

    def __init__(self,
                filepath_field: str,
                drop_filepath_field: bool = True,
                recursive: bool = False,
                **kwargs):

        super().__init__(**kwargs)
        self.filepath_field = filepath_field
        self.drop_filepath_field = drop_filepath_field
        self.recursive = recursive
    
    def _count_files(self, data_entry):
        """
        Count the number of files to be removed.
        """
        filepath = Path(data_entry[self.filepath_field])
        if filepath.is_dir():
            if self.recursive:
                file_counter = Counter(f.suffix for f in filepath.iterdir() if f.is_file())
            else:
                raise IsADirectoryError(f"Directory {filepath} is not empty and recursive is False")
        else:
            file_counter = Counter({filepath.suffix : 1})
        return file_counter

    def prepare(self):
        """
        Prepare the processor by counting the number of files to be removed.
        """
        file_counter = Counter()
        for manifest_chunk in self._chunk_manifest():
            chunk_counts = itertools.chain(
                process_map(
                    self._count_files,
                    manifest_chunk,
                    max_workers=self.max_workers,
                    chunksize=self.chunksize,
                    desc="Counting files to remove",
                )
            )
            for entry_file_counter in chunk_counts:
                file_counter += entry_file_counter

        print(f"Total files to remove: {sum(file_counter.values())}")
        for extension, count in file_counter.items():
            logger.info(f"{extension}\t\t{count}")

    def process_dataset_entry(self, data_entry):
        """
        Remove the file or directory specified in the given field of the data entry.

        Args:
            data_entry (dict): A single input sample from the dataset manifest.

        Returns:
            List[DataEntry]: A single-element list containing the updated entry.
        """
        filepath = data_entry[self.filepath_field]

        # Remove the target path from the filesystem
        if os.path.isdir(filepath):
            if self.recursive:
                shutil.rmtree(filepath)  # Recursively delete directory
            else:
                raise IsADirectoryError(f"Directory {filepath} is not empty and recursive is False")
        else:
            os.remove(filepath)  # Delete a single file

        # Optionally remove the filepath field from the data entry
        if self.drop_filepath_field:
            data_entry.pop(self.filepath_field)

        # Wrap and return the modified entry
        return [DataEntry(data=data_entry)]