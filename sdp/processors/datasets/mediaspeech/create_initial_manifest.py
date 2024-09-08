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
from glob import glob

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.common import ffmpeg_convert


class CreateInitialManifestMediaSpeech(BaseParallelProcessor):
    def __init__(
        self,
        data_dir: str,
        output_audio_dir: str,
        audio_file_extenstion: str = "flac",
        text_file_extenstion: str = "txt",
        target_samplerate: int = 16000,
        target_nchannels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.output_audio_dir = output_audio_dir
        self.audio_file_extenstion = audio_file_extenstion
        self.text_file_extenstion = text_file_extenstion
        self.target_samplerate = target_samplerate
        self.target_nchannels = target_nchannels

    def prepare(self):
        os.makedirs(os.path.join(self.output_audio_dir), exist_ok=True)

    def read_manifest(self):
        data_entries = []
        audio_filepaths = glob(f"{self.data_dir}/*{self.audio_file_extenstion}")

        for audio_filepath in audio_filepaths:
            sample_id = os.path.basename(audio_filepath).split(".")[0]
            text_filepaths = glob(
                f"{self.data_dir}/{sample_id}.{self.text_file_extenstion}"
            )

            if len(text_filepaths) < 1:
                logger.warning(
                    f'Sample "{sample_id}" has no related .{self.text_file_extenstion} files. Skipping'
                )
                continue

            text_filepath = text_filepaths[0]
            if len(text_filepaths) > 1:
                logger.warning(
                    f"Sample \"{sample_id}\" has multiple related .{self.text_file_extenstion} files: {', '.join(text_filepaths)}. \
                               Only first file will be used for parsing - {text_filepaths[0]}, other related .{self.text_file_extenstion} files will be skipped."
                )
            data_entries.append(
                {
                    "sample_id": sample_id,
                    "audio_filepath": audio_filepath,
                    "text_filepath": text_filepath,
                }
            )

        return data_entries

    def process_dataset_entry(self, data_entry: DataEntry):
        data = {}
        sample_id = data_entry["sample_id"]
        # Convert source_audio_filepath to .wav
        data["audio_filepath"] = os.path.join(
            os.path.join(self.output_audio_dir, f"{sample_id}.wav"),
        )

        ffmpeg_convert(
            jpg=data_entry["audio_filepath"],
            wav=data["audio_filepath"],
            ar=self.target_samplerate,
            ac=self.target_nchannels,
        )

        if not os.path.exists(data["audio_filepath"]):
            return []

        text_file = open(data_entry["text_filepath"], "r")
        data["text"] = text_file.read()

        return [DataEntry(data=data)]