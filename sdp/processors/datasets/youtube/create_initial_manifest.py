# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Dict
from glob import glob

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.processors.datasets.youtube.utils import parse_srt, Sample
from sdp.utils.common import ffmpeg_convert

class CreateInitialManifest(BaseParallelProcessor):
    def __init__(
        self,
        data_dir: str,
        output_audio_dir: str,
        audio_file_extenstion: str = ".opus",
        target_samplerate: int = 16000,
        target_nchannels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.output_audio_dir = output_audio_dir
        self.audio_file_extenstion = audio_file_extenstion
        self.target_samplerate = target_samplerate
        self.target_nchannels = target_nchannels

    def _get_manifest(self):
        audio_filepaths = glob(f"{self.data_dir}/*{self.audio_file_extenstion}")
        samples = []
        for audio_filepath in audio_filepaths:
            sample = Sample(orig_audio_filepath = audio_filepath)
            sample.sample_id = os.path.basename(audio_filepath).replace(self.audio_file_extenstion, "") # Get sample_id
            
            # Get .srt file, which relaterd to source audio
            srt_filepaths = glob(f"{self.data_dir}/{sample.sample_id}.*.srt")
            
            if len(srt_filepaths) < 1:
                logger.warning(f"Sample \"{sample.sample_id}\" has no related .srt files. Skipping")
                continue
            
            srt_filepath = srt_filepaths[0]
            if len(srt_filepaths) > 1: 
                logger.warning(f"Sample \"{sample.sample_id}\" has multiple related .srt files: {', '.join(srt_filepaths)}. \
                               Only first file will be used for parsing - {srt_filepaths[0]}, other related .srt files will be skipped.")

            sample.srt_filepath = srt_filepath
            samples.append(sample.to_dataentry())
        
        return samples

    def prepare(self):
        os.makedirs(os.path.join(self.output_audio_dir), exist_ok=True)

    def read_manifest(self):
        data_entries = self._get_manifest()
        return data_entries
    
    def process_dataset_entry(self, data_entry: DataEntry):
        # Convert source_audio_filepath to .wav
        data_entry.data['audio_filepath'] = os.path.join(self.output_audio_dir, os.path.basename(data_entry.data['orig_audio_filepath']).replace(self.audio_file_extenstion, ".wav"))

        ffmpeg_convert(input_file=data_entry.data['orig_audio_filepath'], 
                       output_wav=data_entry.data['audio_filepath'], 
                       sample_rate=self.target_samplerate, 
                       num_channels=self.target_nchannels)

        if not os.path.exists(data_entry.data['audio_filepath']):
            return []
    
        # Parse segments from .srt
        segments = parse_srt(data_entry.data['srt_filepath'], verify_duration = True, wav_filepath=data_entry.data['audio_filepath'])

        if len(segments) > 0:
            data_entry.data['segments'] = [segment.__dict__ for segment in segments]
        
        return [data_entry]