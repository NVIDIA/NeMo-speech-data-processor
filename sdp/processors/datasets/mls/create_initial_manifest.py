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

import os
from pathlib import Path

import sox
from sox import Transformer

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.common import download_file, extract_archive

MLS_URL = "https://dl.fbaipublicfiles.com/mls/mls_{language}.tar.gz"


class CreateInitialManifestMLS(BaseParallelProcessor):
    """Processor to create initial manifest for the Multilingual LibriSpeech (MLS) dataset.

    Dataset link: https://www.openslr.org/94/

    Downloads and unzips raw MLS data for the specified language,
    and creates an initial manifest using the transcripts provided in the raw data.

    Args:
        raw_data_dir (str): the directory where the downloaded data will be/is saved.
            This is also where the extracted and processed data will be.
        language (str): the language of the data you wish to be downloaded.
            This will be used to format the URL from which we attempt to download the data.
            E.g., "english", "italian", "spanish", etc.
        data_split (str): "train", "dev" or "test".
        resampled_audio_dir (str): the directory where the resampled
            wav files will be stored.
        target_samplerate (int): sample rate (Hz) to use for resampling.
            Defaults to 16000.
        target_nchannels (int): number of channels to create during resampling process.
            Defaults to 1.

    Returns:
        This processor generates an initial manifest file with the following fields::

            {
                "audio_filepath": <path to the audio file>,
                "duration": <duration of the audio in seconds>,
                "text": <transcription>,
            }
    """

    def __init__(
        self,
        raw_data_dir: str,
        language: str,
        data_split: str,
        resampled_audio_dir: str,
        target_samplerate: int = 16000,
        target_nchannels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        self.language = language
        self.data_split = data_split
        self.resampled_audio_dir = Path(resampled_audio_dir)
        self.target_samplerate = target_samplerate
        self.target_nchannels = target_nchannels

        # will be initialized in self.prepare method
        self.audio_path_prefix = None
        self.transcription_file = None

    def prepare(self):
        """Downloading and extracting data (unless already done)."""
        os.makedirs(self.raw_data_dir, exist_ok=True)

        url = MLS_URL.format(language=self.language)

        if not (self.raw_data_dir / f"mls_{self.language}.tar.gz").exists():
            download_file(url, str(self.raw_data_dir))

        data_folder = extract_archive(str(self.raw_data_dir / os.path.basename(url)), str(self.raw_data_dir))

        self.audio_path_prefix = str(Path(data_folder) / self.data_split / "audio")
        self.transcription_file = str(Path(data_folder) / self.data_split / "transcripts.txt")

    def read_manifest(self):
        """Reading the initial data line-by-line."""
        if self.transcription_file is None:
            raise RuntimeError("self.process has to be called before processing the data.")

        with open(self.transcription_file, "rt", encoding="utf8") as fin:
            dataset_entries = fin.readlines()

        return dataset_entries

    def process_dataset_entry(self, data_entry: str):
        """Processing the data entries.

        Converts all audio into wav format and outputs filepath, duration and
        transcription text.
        """
        if len(data_entry.split("\t")) != 2:
            raise RuntimeError(f"have more than one tab in line {data_entry}")

        utt_id, text = data_entry.split("\t")
        transcript_text = text.strip()

        src_flac_path = os.path.join(self.audio_path_prefix, *utt_id.split("_")[:2], utt_id + ".flac")
        tgt_wav_path = os.path.join(self.resampled_audio_dir, *utt_id.split("_")[:2], utt_id + ".wav")

        if not os.path.exists(os.path.dirname(tgt_wav_path)):
            os.makedirs(os.path.dirname(tgt_wav_path), exist_ok=True)
        if not os.path.exists(tgt_wav_path):
            tfm = Transformer()
            tfm.rate(samplerate=self.target_samplerate)
            tfm.channels(n_channels=self.target_nchannels)
            tfm.build(input_filepath=src_flac_path, output_filepath=tgt_wav_path)

        data = {
            "audio_filepath": tgt_wav_path,
            "duration": float(sox.file_info.duration(tgt_wav_path)),
            "text": transcript_text,
        }

        return [DataEntry(data=data)]
