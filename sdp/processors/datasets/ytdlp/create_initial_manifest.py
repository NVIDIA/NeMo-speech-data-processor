# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import subprocess
from pathlib import Path
import json
from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class CreateInitialManifestytdlp(BaseParallelProcessor):
    """
    Processor for creating an initial dataset manifest by saving youtube links.
    Make sure to install yt-dlp tool before funning this code. 

    Tool link: https://github.com/yt-dlp/yt-dlp

    Args:
        raw_data_dir (str): Root directory of the files to be added to the manifest. Recursively searches for files with the given 'extension'.
        output_field (str): Field to store the file paths in the dataset. Default is "audio_filepath".
        extension (str): Extension of the files to include in the dataset. Default is "wav".
        **kwargs: Additional keyword arguments for the base class `BaseParallelProcessor`.
    """

    def __init__(
        self,
        raw_data_dir: str,
        output_field: str = "audio_filepath",
        # extension: str = "wav",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        self.output_field = output_field
        file_path = "sdp/processors/datasets/ytdlp/search_terms.json"
        
        with open(file_path, "r") as f:
            channels = json.load(f)

        self.channel_tuples = [(channel["search_term"], channel["audio_count"]) for channel in channels["channels"]]


    def read_manifest(self):
        channels_data = []
        for search_term, audio_count in self.channel_tuples:
            if search_term is not None:
                command = [
                    'yt-dlp',
                    f'ytsearch{audio_count}:{search_term}',
                    '--match-filter', "license = 'Creative Commons Attribution license (reuse allowed)'",
                    '--get-id',
                ]
                try:
                    process = subprocess.run(command, stdout=subprocess.PIPE, text=True)
                    output = process.stdout.strip()
                    # Each video ID will be on a new line, so split the output into a list of IDs
                    video_ids = output.split('\n')
                    while("" in video_ids):
                        video_ids.remove("")
                    # Construct the full YouTube page URL for each video ID
                    youtube_base_url = "https://www.youtube.com/watch?v="
                    # Append the data to the channels_data dictionary
                    logger.info("Got youtube links :", video_ids)
                    channels_data.extend(
                        [(youtube_base_url + video_id, video_id) for video_id in video_ids]
                    )
                except subprocess.CalledProcessError as e:
                    print(f"Error fetching URLs for {search_term}: {e}")
            else:
                continue

        return channels_data
    
    def process_dataset_entry(self, data_entry):
        data = {self.output_field: data_entry[0],'youtube_id':data_entry[1]}
        return [DataEntry(data=data)]
