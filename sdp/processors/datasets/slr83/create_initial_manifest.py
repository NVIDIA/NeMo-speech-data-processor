# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
from pathlib import Path
from typing import Optional

import sox

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.common import download_file, extract_archive

DATASET_URL = "https://www.openslr.org/resources/83/{dialect}.zip"


class CreateInitialManifestSLR83(BaseParallelProcessor):
    """Processor to create initial manifest for the SLR83 dataset.

    This is a dataset introduced in `Open-source Multi-speaker Corpora of the
    English Accents in the British Isles <https://aclanthology.org/2020.lrec-1.804/>`_.

    The original paper does not provide train/dev/test split, so we include a
    custom splits.json file that can be used as a standardized split to compare
    results. For more details on this data split see `Damage Control During
    Domain Adaptation for Transducer Based Automatic Speech Recognition
    <https://arxiv.org/abs/2210.03255>`_.
    """

    AVAILABLE_DIALECTS = [
        'irish_english_male',
        'midlands_english_female',
        'midlands_english_male',
        'northern_english_female',
        'northern_english_male',
        'scottish_english_female',
        'scottish_english_male',
        'southern_english_female',
        'southern_english_male',
        'welsh_english_female',
        'welsh_english_male',
    ]

    def __init__(
        self,
        raw_data_dir: str,
        dialect: str,
        data_split: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        self.dialect = dialect
        if dialect not in self.AVAILABLE_DIALECTS:
            raise ValueError(f"dialect has to be one of {self.AVAILABLE_DIALECTS}")
        self.data_split = data_split
        with open(Path(__file__).parent / 'splits.json', 'rt', encoding="utf-8") as fin:
            self.utt_ids = json.load(fin)[self.dialect]

    def prepare(self):
        """Downloading and extracting data (unless already done)."""
        os.makedirs(self.raw_data_dir, exist_ok=True)

        url = DATASET_URL.format(dialect=self.dialect)

        if not (self.raw_data_dir / f"{self.dialect}.zip").exists():
            download_file(url, str(self.raw_data_dir))

        extract_archive(str(self.raw_data_dir / os.path.basename(url)), str(self.raw_data_dir))
        self.transcription_file = str(self.raw_data_dir / "line_index.csv")

    def read_manifest(self):
        if self.transcription_file is None:
            raise RuntimeError("self.process has to be called before processing the data.")

        with open(self.transcription_file, "rt", encoding="utf8") as fin:
            dataset_entries = fin.readlines()

        return dataset_entries

    def process_dataset_entry(self, data_entry: str):
        split_entry = data_entry.split(", ")
        if len(split_entry) != 3:
            raise RuntimeError(f"Input data is badly formatted! Bad line: {data_entry}")

        _, utt_id, transcript_text = split_entry
        if self.data_split is not None and utt_id not in self.utt_ids[self.data_split]:
            return []
        audio_path = str(self.raw_data_dir / (utt_id + ".wav"))
        data = {
            "audio_filepath": audio_path,
            "duration": float(sox.file_info.duration(audio_path)),
            "text": transcript_text.strip(),
        }

        return [DataEntry(data=data)]
