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
from pathlib import Path

from sox import Transformer

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.common import download_file, extract_archive

DATASET_URL = "https://www.openslr.org/resources/102/ISSAI_KSC_335RS_v1.1_flac.tar.gz"


class CreateInitialManifestSLR102(BaseParallelProcessor):
    """Processor to create initial manifest for the Kazakh Speech Corpus (KSC) / OpenSLR102 dataset.

    Dataset link: https://www.openslr.org/resources/102/

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
        This processor generates an initial manifest file with the following fields::

            {
                "audio_filepath": <path to the audio file>,
                "text": <transcription (with capitalization and punctuation)>,
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
            download_file(DATASET_URL, self.raw_data_dir)

        elif len(tar_gz_files) > 1:
            raise RuntimeError(f"Expecting exactly one *.tar.gz file in directory {self.raw_data_dir}")

        data_folder = extract_archive(tar_gz_files[0], self.extract_archive_dir)

        self.audio_path_prefix = Path(data_folder, "Audios_flac")
        self.transcription_path_prefix = Path(data_folder, "Transcriptions")
        self.transcription_path_file = Path(data_folder, "Meta", self.data_split).with_suffix(".csv")

        os.makedirs(self.resampled_audio_dir, exist_ok=True)

    def read_manifest(self):
        if self.transcription_path_file is None:
            raise RuntimeError("self.process has to be called before processing the data.")

        with open(self.transcription_path_file, "rt", encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=" ")
            next(reader, None)  # skip the headers
            dataset_entries = [row["uttID"] for row in reader]
        return dataset_entries

    def process_dataset_entry(self, utt_id: str):
        with open(Path(self.transcription_path_prefix, utt_id).with_suffix(".txt"), "rt") as txtfile:
            transcript_text = " ".join(txtfile.readlines()).strip()

        audio_path = Path(self.audio_path_prefix, utt_id).with_suffix(".flac")
        output_wav_path = Path(self.resampled_audio_dir, utt_id).with_suffix(".wav")

        if not os.path.exists(output_wav_path):
            tfm = Transformer()
            tfm.rate(samplerate=self.target_samplerate)
            tfm.channels(n_channels=self.target_nchannels)
            tfm.build(input_filepath=audio_path, output_filepath=output_wav_path)

        data = {
            "audio_filepath": output_wav_path.as_posix(),
            "text": transcript_text,
        }

        return [DataEntry(data=data)]
