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

from sdp.processors.base_processor import DataEntry, BaseParallelProcessor
from sdp.processors import ListToEntries
from sdp.logging import logger


class CreateInitialManifestYodas2(ListToEntries):
    """
    Custom processor for generating an initial manifest from a dataset entry where
    one of the fields contains a list of segment dictionaries.

    In addition to flattening entries via ListToEntries, it also:
    - Extracts `yodas_id` from each entry's audio file name.
    - Enriches each entry with a precomputed `duration` from an external duration file.

    The final output will contain entries with the following required fields:
        - source_audio_filepath
        - yodas_id
        - duration

    Args:
        **kwargs: Passed through to ListToEntries, including:
            - field_with_list (str): Name of the field to flatten.
            - output_field (str): Required if inner list items are primitives.
            - fields_to_save (list[str]): Fields to preserve from the original entry.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_samples_durations(self, durations_filepath: str):
        """
        Loads a mapping of yodas_id → duration from a plain text file.

        The file should contain one line per sample, formatted as:
        <yodas_id> <duration>

        Args:
            durations_filepath (str): Path to the durations metadata file.

        Returns:
            dict[str, float]: Mapping from sample ID to duration.
        """
        durations = dict()
        with open(durations_filepath, 'r') as durations_txt:
            for line in durations_txt:
                yodas_id, duration = line.strip().split()
                durations[yodas_id] = float(duration)
        return durations

    def process_dataset_entry(self, data_entry):
        """
        Processes a single dataset entry:
        - Loads durations for all samples listed in the `local_duration` file.
        - Flattens the input using the ListToEntries logic.
        - Attaches `yodas_id` (from file basename) and `duration` (from metadata file) to each entry.
        - Skips entries not present in the durations file, with a warning.

        Args:
            data_entry (dict): One entry from the input manifest, must contain the
                               'local_duration' field with path to the durations file.

        Returns:
            list[DataEntry]: Flattened and enriched list of manifest entries.
        """
        # Load durations for all samples
        durations = self.get_samples_durations(data_entry['local_duration'])

        # Flatten the input field with list elements using base ListToEntries
        data_entries = super().process_dataset_entry(data_entry)

        yodas_entries = []
        for entry in data_entries:
            # Extract yodas_id from audio filename (without extension)
            yodas_id = os.path.basename(entry.data['source_audio_filepath']).split('.')[0]
            entry.data['yodas_id'] = yodas_id

            # Add duration if available
            if yodas_id in durations:
                entry.data['duration'] = durations[yodas_id]
                yodas_entries.append(entry)
            else:
                logger.warning(f'Skipping `{yodas_id}` because there is no duration info in metadata.')

        return yodas_entries
