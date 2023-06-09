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
import subprocess
from pathlib import Path

import sox
from sox import Transformer

from sdp.logging import logger
from sdp.processors.base_processor import BaseParallelProcessor, DataEntry

VOXPOPULI_URL = "https://github.com/facebookresearch/voxpopuli"


class CreateInitialManifestVoxpopuli(BaseParallelProcessor):
    """
    Downloads and unzips raw VoxPopuli data for the specified language,
    and creates an initial manifest using the transcripts provided in the
    raw data.

    Args:
        raw_data_dir: the directory where the downloaded data will be/is saved.
        language_id: the language of the data you wish to be downloaded.
        data_split: the data split for which the initial manifest will
            be created.
        resampled_audio_dir: the directory where the resampled (16kHz) wav
            files will be stored.
    """

    def __init__(
        self,
        raw_data_dir: str,
        language_id: str,
        data_split: str,
        resampled_audio_dir: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        self.language_id = language_id
        self.data_split = data_split
        self.resampled_audio_dir = resampled_audio_dir

    def prepare(self):
        """Downloading data (unless already done)"""
        os.makedirs(self.raw_data_dir, exist_ok=True)

        if not (self.raw_data_dir / "transcribed_data").exists():
            # TODO: some kind of isolated environment?
            if not os.path.exists(self.raw_data_dir / 'voxpopuli'):
                logger.info("Downloading voxpopuli and installing requirements")
                subprocess.run(f"git clone {VOXPOPULI_URL} {self.raw_data_dir / 'voxpopuli'}", check=True, shell=True)
                subprocess.run(
                    f"pip install -r {self.raw_data_dir / 'voxpopuli' / 'requirements.txt'}", check=True, shell=True
                )
            if not os.path.exists(self.raw_data_dir / 'raw_audios'):
                logger.info("Downloading raw audios")
                subprocess.run(
                    f"cd {self.raw_data_dir / 'voxpopuli'} && "
                    f"python -m voxpopuli.download_audios --root {self.raw_data_dir} --subset asr",
                    check=True,
                    shell=True,
                )
            if not os.path.exists(self.raw_data_dir / 'transcribed_data' / self.language_id):
                logger.info("Segmenting and transcribing the data")
                subprocess.run(
                    f"cd {self.raw_data_dir / 'voxpopuli'} && "
                    f"python -m voxpopuli.get_asr_data  --root {self.raw_data_dir} --lang {self.language_id}",
                    check=True,
                    shell=True,
                )

    def read_manifest(self):
        with open(
            self.raw_data_dir / "transcribed_data" / self.language_id / f"asr_{self.data_split}.tsv",
            "rt",
            encoding="utf8",
        ) as fin:
            dataset_entries = fin.readlines()[1:]  # skip header line

        return dataset_entries

    def process_dataset_entry(self, data_entry: str):
        if len(data_entry.split("\t")) != 8:
            raise RuntimeError(f"have more/less than 7 tabs in line {data_entry}")

        utt_id, raw_text, norm_text, spk_id, _, gender, is_gold_transcript, accent = data_entry.split("\t")
        year = utt_id[:4]

        src_audio_path = os.path.join(self.raw_data_dir, "transcribed_data", self.language_id, year, utt_id + ".ogg")
        tgt_wav_path = os.path.join(self.resampled_audio_dir, utt_id + ".wav")

        if not os.path.exists(os.path.dirname(tgt_wav_path)):
            os.makedirs(os.path.dirname(tgt_wav_path), exist_ok=True)
        if not os.path.exists(tgt_wav_path):
            Transformer().build(src_audio_path, tgt_wav_path)

        data = {
            "audio_filepath": tgt_wav_path,
            "duration": float(sox.file_info.duration(tgt_wav_path)),
            "text": norm_text,
            "provided_norm_text": norm_text,
            "raw_text": raw_text,
            "spk_id": spk_id,
            "gender": gender,
            "is_gold_transcript": is_gold_transcript,
            "accent": accent,
        }
        return [DataEntry(data=data)]
