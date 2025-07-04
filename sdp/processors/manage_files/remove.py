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
                **kwargs):

        super().__init__(**kwargs)
        self.filepath_field = filepath_field
        self.drop_filepath_field = drop_filepath_field

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
            shutil.rmtree(filepath)  # Recursively delete directory
        else:
            os.remove(filepath)  # Delete a single file

        # Optionally remove the filepath field from the data entry
        if self.drop_filepath_field:
            data_entry.pop(self.filepath_field)

        # Wrap and return the modified entry
        return [DataEntry(data=data_entry)]