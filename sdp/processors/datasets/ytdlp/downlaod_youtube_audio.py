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
import os
from pathlib import Path
import json
from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry



class GetYoutubeAudio(BaseParallelProcessor):
    """
    Processor to download audio from YouTube links and calculate the duration of the audio.

    Args:
        links_filepath_field (str): Field to get the YouTube video link.
        output_audio_path (str): Path to save the downloaded audio files.
        **kwargs: Additional keyword arguments for the base class `BaseParallelProcessor`.
    
    Returns:
        All the same fields as in the input manifest plus the audio duration.
    """
    def __init__(
        self,
        links_filepath_field: str,
        output_audio_path: str,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.links_filepath_field = links_filepath_field
        self.output_audio_path = output_audio_path
        path = Path(output_audio_path)
        path.mkdir(exist_ok=True)

    def process_dataset_entry(self, data_entry):
        audio_link = data_entry[self.links_filepath_field]
        logger.info(f"Processing audio link: {audio_link}")
        output_path = os.path.join(self.output_audio_path, data_entry['youtube_id'] + '.wav')
        
        os.makedirs(self.output_audio_path, exist_ok=True)

        if not os.path.exists(output_path):
            # Download audio with postprocessor sample rate = 16k
            command = f'yt-dlp -x --audio-format wav --postprocessor-args "-ac 1 -ar 16000" -o "{output_path}" "{audio_link}"'
            try:
                subprocess.run(command, shell=True, check=True)
                logger.info(f"Audio downloaded successfully: {output_path}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to download audio: {e}")
        else:
            logger.info(f"Output file already exists: {output_path}")

        ffprobe_cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{output_path}"'
        try:
            duration_str = subprocess.run(ffprobe_cmd, shell=True, check=True, stdout=subprocess.PIPE, text=True).stdout.strip()
            duration = float(duration_str)
            logger.info(f"Audio length: {duration} seconds")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to get audio duration: {e}")
            duration = None  

        data = {
            self.links_filepath_field: output_path,
            'youtube_id': data_entry['youtube_id'],
            'duration': duration 
        }
        return [DataEntry(data=data)]