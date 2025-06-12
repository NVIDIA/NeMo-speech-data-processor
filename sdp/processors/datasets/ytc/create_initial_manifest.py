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
import subprocess

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.common import load_manifest

class CreateInitialManifestYTC(BaseParallelProcessor):
    """A processor class for creating initial manifest files for a TTS dataset.
    
    It takes a manifest file containing audio file paths and resamples them to a target
    sample rate and format, while creating a new manifest file with the updated paths.

    Args:
        input_format (str): Format of the input audio files
        resampled_audio_dir (str): Directory where resampled audio files will be saved
        target_sample_rate (int): Desired sample rate for the output audio files
        target_format (str): Desired format for the output audio files
        target_nchannels (int): Desired number of channels for the output audio files

    Returns:
        The same data as in the input manifest, but with resampled audio files and
        updated audio file paths.

    Example:
        .. code-block:: yaml

            - _target_: sdp.processors.datasets.ytc.create_initial_manifest.CreateInitialManifestYTC
              input_manifest_file: ${workspace_dir}/manifest.json
              output_manifest_file: ${workspace_dir}/manifest_resampled.json
    """
    def __init__(
            self,
            input_format: str,
            resampled_audio_dir: str,
            target_sample_rate: int,
            target_format: str,
            target_nchannels: int,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.input_format = input_format
        self.resampled_audio_dir = resampled_audio_dir
        self.target_sample_rate = target_sample_rate
        self.target_format = target_format
        self.target_nchannels = target_nchannels

    def prepare(self):
        """Creates the output directory for resampled audio files if it doesn't exist."""
        os.makedirs(self.resampled_audio_dir, exist_ok=True)

    def read_manifest(self):
        """ Reads metadata from JSONL file in the input manifest
          Returns:
            list: A list of dataset entries parsed from the JSONL manifest file
        """
        dataset_entries = load_manifest(self.input_manifest_file, encoding="utf8")

        return dataset_entries

    def process_dataset_entry(self, metadata: DataEntry):
        """Processes a single dataset entry by resampling the audio file and updating metadata.
        
        Args:
            metadata (DataEntry): The metadata entry containing information about the audio file
            
        Returns:
            list[DataEntry]: A list containing the processed DataEntry with updated metadata
            
        Note:
            This method:
            1. Resamples the audio file to the target format and sample rate if needed
            2. Updates the metadata with new file paths and duration
            3. Uses either sox or ffmpeg for audio conversion depending on input format
        """
        import soundfile as sf
        input_audio_path = metadata['audio_filepath']
        output_audio_path = os.path.join(self.resampled_audio_dir, metadata['audio_item_id'] + '.' + self.target_format)

        # Convert audio file to target sample rate and format
        if not os.path.exists(output_audio_path):
            if input_audio_path.lower().endswith(".wav"):
                cmd = f'sox --no-dither -V1 "{input_audio_path}" -r {self.target_sample_rate} -c 1 -b 16 "{output_audio_path}"'
            else:
                cmd = f'ffmpeg -i  "{input_audio_path}" -ar {self.target_sample_rate} -ac 1 -ab 16 "{output_audio_path}" -v error'
            try:
                subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True)  # Ensures output is in string formats)
            except subprocess.CalledProcessError as e:
                print("Exception occurred while converting audio file: ", e, e.stderr)
                print(f'Error converting {input_audio_path} to {output_audio_path}. Hence skipping this entry.')
                exit(1)
        
        metadata['audio_filepath'] = input_audio_path
        metadata['resampled_audio_filepath'] = output_audio_path
        try:
            metadata['duration'] = sf.info(output_audio_path).duration
        except Exception as e:
            print(f'Error getting duration of {output_audio_path}. Hence not adding duration to metadata.')
        
        return [DataEntry(data=metadata)]


