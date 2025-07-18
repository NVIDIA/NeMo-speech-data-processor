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

import tarfile
import os
from pathlib import Path

from sdp.logging import logger
from sdp.processors.base_processor import DataEntry, BaseParallelProcessor


class ExtractTar(BaseParallelProcessor):
    """
    A processor that extracts `.tar` archives for each entry in a dataset.

    This processor reads a filepath to a tar archive from a specific field in the dataset entry,
    extracts the contents into a specified directory, and optionally appends the extracted file paths
    or the extraction directory to the entry under a new field.

    Args:
        field_to_tar_filepath (str): The field in the input entry that contains the path to the `.tar` file.
        extraction_dir (str): The base directory where extracted files should be placed.
        remove_source_tar (bool): If True, deletes the original `.tar` file after successful extraction.
        skip_invalid_filepaths (bool): If True, logs and skips invalid paths instead of raising exceptions.
        filepath_prefix_field (str): Optional field in the entry used as a subdirectory prefix under `extraction_dir`.
        output_filepath_field (str): Field name where the output (path or list of paths) will be stored.
        get_extracted_filepaths (bool): If True, collects and returns a list of all extracted file paths.

    Returns:
        A manifest where each entry is updated with the path to the extracted files or directory.
    """

    def __init__(
        self, 
        field_to_tar_filepath: str, 
        extraction_dir: str, 
        remove_source_tar: bool = False, 
        skip_invalid_filepaths: bool = False,
        filepath_prefix_field: str = None,
        output_filepath_field: str = 'extracted',
        get_extracted_filepaths: bool = False, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.field_to_tar_filepath = field_to_tar_filepath
        self.extraction_dir = extraction_dir
        self.remove_source_tar = remove_source_tar
        self.skip_invalid_filepaths = skip_invalid_filepaths
        self.filepath_prefix_field = filepath_prefix_field
        self.output_filepath_field = output_filepath_field
        self.get_extracted_filepaths = get_extracted_filepaths

    def process_dataset_entry(self, data_entry):
        # Read the tar file path from the specified field
        tar_filepath = data_entry[self.field_to_tar_filepath]

        # Handle missing or invalid tar file path
        if not isinstance(tar_filepath, str) or not os.path.exists(tar_filepath):
            if self.skip_invalid_filepaths:
                logger.info(f"Invalid filepath {tar_filepath}. Skipping..")
                output_filepath = None
            else:
                raise ValueError(f"Invalid filepath {tar_filepath}.")
        else:
            # Determine output path using optional prefix and tar filename
            output_filepath_prefix = (
                data_entry[self.filepath_prefix_field]
                if self.filepath_prefix_field and data_entry.get(self.filepath_prefix_field)
                else ''
            )
            output_filepath = os.path.join(
                self.extraction_dir,
                output_filepath_prefix,
                os.path.basename(tar_filepath).split('.')[0]
            )
            os.makedirs(output_filepath, exist_ok=True)

            # Extract tar archive into target directory
            try:
                with tarfile.open(tar_filepath, 'r') as tar:
                    tar.extractall(path=output_filepath)
            except Exception as e:
                if self.skip_invalid_filepaths:
                    logger.info(f"Error extracting {tar_filepath}: {e}. Skipping..")
                    output_filepath = None
                else:
                    raise ValueError(f"Error extracting {tar_filepath}: {e}")

            # Gather list of all extracted files if requested
            extracted_filepaths = []
            if output_filepath is not None and self.get_extracted_filepaths:
                extraction_folder_path = Path(output_filepath)
                extracted_filepaths = [
                    str(file) for file in extraction_folder_path.rglob("*") if file.is_file()
                ]

            # Optionally remove the original tar archive after extraction
            if self.remove_source_tar:
                os.remove(tar_filepath)

        # Write the extraction result into the entry (either path or file list)
        if self.get_extracted_filepaths:
            data_entry[self.output_filepath_field] = extracted_filepaths
        else:
            data_entry[self.output_filepath_field] = output_filepath

        return [DataEntry(data=data_entry)]