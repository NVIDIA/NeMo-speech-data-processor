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

# To convert mp3 files to wav using sox, you must have installed sox with mp3 support
# For example sudo apt-get install libsox-fmt-mp3
import csv
import glob
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

from sox import Transformer
from tqdm.contrib.concurrent import process_map

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.common import download_file, extract_archive


class CreateInitialManifestKSC2(BaseParallelProcessor):
    """Processor to create initial manifest for the Kazakh Speech Corpus (KSC) 2.

    The dataset should be requested via Google Forms, which can be found here https://issai.nu.edu.kz/kz-speech-corpus/.

    Extracts raw data for the specified language and creates an initial manifest
    using the transcripts provided in the raw data.

    Args:
        raw_data_dir (str): the path to the directory containing the raw data archive file.
        extract_archive_dir (str): directory where the extracted data will be saved.
        resampled_audio_dir (str): directory where the resampled audio will be saved.
        data_split (str): "train", "dev" or "test".
        target_samplerate (int): sample rate (Hz) to use for resampling.
            Defaults to 16000.
        target_nchannels (int): number of channels to create during resampling process.
            Defaults to 1.
    Returns:
        This processor generates an initial manifest file with the following fields:

            {
                "audio_filepath": <path to the audio file>,
                "text": <transcription (with capitalization and punctuation)>,
                "source": <source of the given data>,
            }
    """

    def __init__(
        self,
        raw_data_dir: str,
        extract_archive_dir: str,
        resampled_audio_dir: str,
        data_split: str,
        target_samplerate: int = 16000,
        target_nchannels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        self.extract_archive_dir = extract_archive_dir
        self.resampled_audio_dir = resampled_audio_dir
        self.data_split = data_split
        self.target_samplerate = target_samplerate
        self.target_nchannels = target_nchannels

    def prepare(self):
        """Extracting data (unless already done)."""
        os.makedirs(self.raw_data_dir, exist_ok=True)

        tar_gz_files = glob.glob(str(self.raw_data_dir) + f"/*.tar.gz")

        if not tar_gz_files:
            raise RuntimeError(
                f"Did not find any file matching {self.raw_data_dir}/*.tar.gz. "
                "For KSC2 dataset we cannot automatically download the data, so "
                "make sure to get the data manually"
                "and put it in the 'raw_data_dir' folder."
            )

        elif len(tar_gz_files) > 1:
            raise RuntimeError(f"Expecting exactly one *.tar.gz file in directory {self.raw_data_dir}")

        data_folder = extract_archive(tar_gz_files[0], self.extract_archive_dir)

        if self.data_split.capitalize() not in data_folder:
            self.data_split_dir = Path(data_folder, self.data_split.capitalize())
        else:
            self.data_split_dir = Path(data_folder)

        os.makedirs(self.resampled_audio_dir, exist_ok=True)

    def read_manifest(self):
        if self.data_split_dir is None:
            raise RuntimeError("self.process has to be called before processing the data.")

        dataset_entries = []

        without_text = defaultdict(int)

        for audio_filepath in self.data_split_dir.rglob('*.flac'):
            filename = audio_filepath.stem
            source = audio_filepath.relative_to(self.data_split_dir).parents[0].as_posix()

            transcribed_filename = Path(audio_filepath.parent, filename).with_suffix('.txt')

            if transcribed_filename.exists():
                with open(transcribed_filename, "rt", encoding="utf8") as txtfile:
                    text = ' '.join(txtfile.readlines())
            elif transcribed_filename.with_suffix('.txt.txt').exists():
                transcribed_filename = transcribed_filename.with_suffix('.txt.txt')
                with open(transcribed_filename, "rt", encoding="utf8") as txtfile:
                    text = ' '.join(txtfile.readlines())
            else:
                without_text[audio_filepath.parent] += 1
                continue

            entry = {'audio_filepath': audio_filepath.as_posix(), 'text': text, 'source': source}

            dataset_entries.append(entry)

        logger.info(f"Without text entries -> {without_text}")

        return dataset_entries

    def process_dataset_entry(self, data_entry: Dict):
        wav_source_dir = Path(self.resampled_audio_dir, data_entry['source'])
        wav_source_dir.mkdir(exist_ok=True)

        output_wav_path = Path(wav_source_dir, Path(data_entry['audio_filepath']).stem).with_suffix(".wav")

        if not os.path.exists(output_wav_path):
            tfm = Transformer()
            tfm.rate(samplerate=self.target_samplerate)
            tfm.channels(n_channels=self.target_nchannels)
            tfm.build(input_filepath=data_entry['audio_filepath'], output_filepath=output_wav_path)

        data_entry['audio_filepath'] = output_wav_path.as_posix()

        return [DataEntry(data=data_entry)]
