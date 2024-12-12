# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from pathlib import Path

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.common import ffmpeg_convert
from sdp.utils.common import extract_archive


class CreateInitialManifestMediaSpeech(BaseParallelProcessor):
    """
    Processor for creating initial manifest for MediaSpeech Arabic dataset.
    Dataset link: https://www.openslr.org/108/.
    Prior to calling processor download the tarred dataset and store it under `raw_dataset_dir/AR.tgz` or `raw_dataset_dir/AR.tar.gz`.
    
    Args:
        raw_data_dir (str): The root directory of the dataset.
        extract_archive_dir (str): Directory where the extracted data will be saved.
        resampled_audios_dir (str): Directory where the resampled audio will be saved.
        already_extracted (bool): If True, we will not try to extract the raw data. Defaults to False.
        target_samplerate (int): Sample rate (Hz) to use for resampling. Defaults to 16000.
        target_nchannels (int): Number of channels to create during resampling process. Defaults to 1.
        output_manifest_sample_id_key (str): The field name to store sample ID. Defaults to "sample_id".
        output_manifest_audio_filapath_key (str): The field name to store audio file path. Defaults to "audio_filepath".
        output_manifest_text_key (str): The field name to store text. Defaults to "text".
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    Returns:
        This processor generates an initial manifest file with the following fields::
        
            {
                "audio_filepath": <path to the audio file>,
                "text": <text>,
            }
    """
    def __init__(
        self,
        raw_data_dir: str,
        resampled_audios_dir: str,
        extract_archive_dir: str,
        already_extracted: bool = False,
        target_samplerate: int = 16000,
        target_nchannels: int = 1,
        output_manifest_sample_id_key: str = "sample_id",
        output_manifest_audio_filapath_key: str = "audio_filepath",
        output_manifest_text_key: str = "text",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        self.extract_archive_dir = extract_archive_dir
        self.resampled_audios_dir = Path(resampled_audios_dir)
        self.already_extracted = already_extracted
        
        self.target_samplerate = target_samplerate
        self.target_nchannels = target_nchannels
        
        self.output_manifest_sample_id_key = output_manifest_sample_id_key
        self.output_manifest_audio_filapath_key = output_manifest_audio_filapath_key
        self.output_manifest_text_key = output_manifest_text_key

    def prepare(self):
        # Extracting data (unless already done).
        if not self.already_extracted:
            tar_gz_filepath = Path(str(self.raw_data_dir)) / "AR.tgz"
            if not tar_gz_filepath.exists():
                # necessary to check in tests
                tar_gz_filepath = Path(str(self.raw_data_dir)) / "AR.tar.gz"

            if not tar_gz_filepath.exists():
                raise RuntimeError(
                    '''Did not find any file matching `AR.tgz` or `AR.tar.gz`. 
                    For MediaSpeech dataset we cannot automatically download the data, so 
                    make sure to get the data from https://www.openslr.org/108/ 
                    and put it in the `raw_data_dir` folder.'''
                )

            self.dataset_dir = Path(extract_archive(tar_gz_filepath, self.extract_archive_dir))
        else:
            logger.info("Skipping dataset untarring...")
            self.dataset_dir = Path(self.extract_archive_dir) / "AR"
        
        os.makedirs(self.resampled_audios_dir, exist_ok=True)

    def read_manifest(self):
        data_entries = []
        audio_filepaths = glob(f"{self.dataset_dir}/*.flac")

        for audio_filepath in audio_filepaths:
            sample_id = os.path.basename(audio_filepath).split(".")[0]

            text_filepath = f"{self.dataset_dir}/{sample_id}.txt"
            if not os.path.exists(text_filepath):
                logger.warning(
                    f'Sample "{sample_id}" has no related .txt files. Skipping'
                )
                continue

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
        data[self.output_manifest_audio_filapath_key] = os.path.join(
            os.path.join(self.resampled_audios_dir, f"{sample_id}.wav"),
        )

        ffmpeg_convert(
            jpg=data_entry["audio_filepath"],
            wav=data[self.output_manifest_audio_filapath_key],
            ar=self.target_samplerate,
            ac=self.target_nchannels,
        )

        if not os.path.exists(data[self.output_manifest_audio_filapath_key]):
            return []

        text_file = open(data_entry["text_filepath"], "r")
        data[self.output_manifest_text_key] = text_file.read()

        return [DataEntry(data=data)]