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
from nemo.utils import logging
from sox import Transformer

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.common import extract_archive

VOXPOPULI_URL = "https://github.com/facebookresearch/voxpopuli"


class CreateInitialManifestVoxpopuli(BaseParallelProcessor):
    """
    Downloads and unzips raw VoxPopuli data for the specified language,
    and creates an initial manifest using the transcripts provided in the
    raw data.

    Args:
        language_id: the language of the data you wish to be downloaded.
        download_dir: the directory where the downloaded data will be saved.
        data_split: the data split for which the initial manifest will
            be created.
        resampled_audio_dir: the directory where the resampled (16kHz) wav
            files will be stored.
        use_test_data: if `True`, will use the test data manifest located
            at `TEST_DATA_PATH` to carry out tests.
    """

    def __init__(
        self,
        language_id: str,
        download_dir: str,
        resampled_audio_dir: str,
        data_split: str,
        use_test_data: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.language_id = language_id
        self.download_dir = Path(download_dir)
        self.data_split = data_split
        self.resampled_audio_dir = resampled_audio_dir
        self.use_test_data = use_test_data

    def prepare(self):
        """Downloading and extracting data (unless already done).

        If use_test_data is True, then will not download data, instead will
        copy the included test data (mainly useful for quick development or
        CI pipeline).
        """
        if self.use_test_data:
            try:
                __TEST_DATA_ROOT = os.environ["TEST_DATA_ROOT"]
                logging.info(f"Found 'TEST_DATA_ROOT' environment variable:{repr(__TEST_DATA_ROOT)}")
            except KeyError:
                raise KeyError(
                    f"Tried to look for os.environ['TEST_DATA_ROOT'] but it was not set."
                    f" Please set 'TEST_DATA_ROOT' as an environment variable and try again."
                )

            self.test_data_path = str(Path(__TEST_DATA_ROOT) / self.language / "voxpopuli" / "data.tar.gz")

            if not os.path.exists(self.test_data_path):
                raise ValueError(
                    f"No such file {self.test_data_path}. Are you sure you specified the "
                    f" 'TEST_DATA_ROOT' environment variable correctly?"
                )
            data_folder = extract_archive(str(self.test_data_path), str(self.download_dir))
        # else:
        # TODO: some kind of isolated environment?
        if not os.path.exists(self.download_dir / 'voxpopuli'):
            logging.info("Downloading voxpopuli and installing requirements")
            subprocess.run(f"git clone {VOXPOPULI_URL} {self.download_dir / 'voxpopuli'}", check=True, shell=True)
            subprocess.run(
                f"pip install -r {self.download_dir / 'voxpopuli' / 'requirements.txt'}", check=True, shell=True
            )
        if not os.path.exists(self.download_dir / 'raw_audios'):
            logging.info("Downloading raw audios")
            subprocess.run(
                f"cd {self.download_dir / 'voxpopuli'} && python -m voxpopuli.download_audios --root {self.download_dir} --subset asr",
                check=True,
                shell=True,
            )
        if not os.path.exists(self.download_dir / 'transcribed_data' / self.language_id):
            logging.info("Segmenting and transcribing the data")
            subprocess.run(
                f"cd {self.download_dir / 'voxpopuli'} && python -m voxpopuli.get_asr_data  --root {self.download_dir} --lang {self.language_id}",
                check=True,
                shell=True,
            )

    def read_manifest(self):
        with open(
            self.download_dir / "transcribed_data" / self.language_id / f"asr_{self.data_split}.tsv",
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

        src_flac_path = os.path.join(self.download_dir, "transcribed_data", self.language_id, year, utt_id + ".ogg")
        tgt_wav_path = os.path.join(self.resampled_audio_dir, utt_id + ".wav")

        if not os.path.exists(os.path.dirname(tgt_wav_path)):
            os.makedirs(os.path.dirname(tgt_wav_path), exist_ok=True)
        if not os.path.exists(tgt_wav_path):
            Transformer().build(src_flac_path, tgt_wav_path)

        data = {
            "audio_filepath": tgt_wav_path,
            "duration": float(sox.file_info.duration(tgt_wav_path)),
            "text": raw_text,
            "provided_norm_text": norm_text,
            "raw_text": raw_text,
            "spk_id": spk_id,
            "gender": gender,
            "is_gold_transcript": is_gold_transcript,
            "accent": accent,
        }
        return [DataEntry(data=data)]
