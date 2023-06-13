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
from typing import Dict, List, Optional, Tuple

import numpy as np
import sox
from tqdm import tqdm

from sdp.logging import logger
from sdp.processors.base_processor import (
    BaseParallelProcessor,
    BaseProcessor,
    DataEntry,
)
from sdp.utils.common import download_file, extract_archive

DATASET_URL = "https://www.openslr.org/resources/83/{dialect}.zip"

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

EXPECTED_SPLIT_STATS = {
    ('irish_english_male', 'test'): (102, 604.757),
    ('irish_english_male', 'train'): (293, 1656.917),
    ('irish_english_male', 'dev'): (53, 302.763),
    ('midlands_english_female', 'test'): (90, 608.341),
    ('midlands_english_female', 'train'): (94, 636.843),
    ('midlands_english_female', 'dev'): (45, 306.261),
    ('midlands_english_male', 'test'): (106, 604.672),
    ('midlands_english_male', 'train'): (270, 1568.683),
    ('midlands_english_male', 'dev'): (52, 301.227),
    ('northern_english_female', 'test'): (267, 1803.435),
    ('northern_english_female', 'train'): (330, 2146.816),
    ('northern_english_female', 'dev'): (145, 906.496),
    ('northern_english_male', 'test'): (587, 3607.467),
    ('northern_english_male', 'train'): (1126, 7003.136),
    ('northern_english_male', 'dev'): (298, 1807.957),
    ('scottish_english_female', 'test'): (284, 1801.301),
    ('scottish_english_female', 'train'): (426, 2681.344),
    ('scottish_english_female', 'dev'): (142, 906.24),
    ('scottish_english_male', 'test'): (612, 3603.883),
    ('scottish_english_male', 'train'): (663, 3994.027),
    ('scottish_english_male', 'dev'): (306, 1800.96),
    ('southern_english_female', 'test'): (572, 3600.128),
    ('southern_english_female', 'train'): (3124, 19213.312),
    ('southern_english_female', 'dev'): (293, 1804.8),
    ('southern_english_male', 'test'): (582, 3600.555),
    ('southern_english_male', 'train'): (3295, 20210.773),
    ('southern_english_male', 'dev'): (296, 1807.445),
    ('welsh_english_female', 'test'): (239, 1805.739),
    ('welsh_english_female', 'train'): (774, 5621.675),
    ('welsh_english_female', 'dev'): (125, 905.387),
    ('welsh_english_male', 'test'): (557, 3605.931),
    ('welsh_english_male', 'train'): (726, 4660.651),
    ('welsh_english_male', 'dev'): (286, 1805.909),
}


class CreateInitialManifestSLR83(BaseParallelProcessor):
    """Processor to create initial manifest for the SLR83 dataset.

    This is a dataset introduced in `Open-source Multi-speaker Corpora of the
    English Accents in the British Isles <https://aclanthology.org/2020.lrec-1.804/>`_.
    """

    def __init__(
        self,
        raw_data_dir: str,
        dialect: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        self.dialect = dialect
        if dialect not in AVAILABLE_DIALECTS:
            raise ValueError(f"dialect has to be one of {AVAILABLE_DIALECTS}")

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
        audio_path = str(self.raw_data_dir / (utt_id + ".wav"))
        data = {
            "audio_filepath": audio_path,
            "duration": float(sox.file_info.duration(audio_path)),
            "text": transcript_text.strip(),
        }

        return [DataEntry(data=data)]


class CustomDataSplitSLR83(BaseProcessor):
    """Split data into train/dev/test.

    The original paper does not provide train/dev/test split, so we include a
    custom processing that can be used as a standardized split to compare
    results. For more details on this data split see `Damage Control During
    Domain Adaptation for Transducer Based Automatic Speech Recognition
    <https://arxiv.org/abs/2210.03255>`_.

    ..note::
        All data dropping has to be done before the split. We will check the
        total number of files to be what is expected in the reference split.
        But if you add any custom pre-processing that changes duration or
        number of files, your splits will likely be different.
    """

    def __init__(self, dialect, data_split, **kwargs):
        super().__init__(**kwargs)
        self.dialect = dialect
        self.data_split = data_split

    def process(self):
        with open(self.input_manifest_file, "rt", encoding="utf8") as fin:
            manifest_data = [json.loads(line) for line in fin.readlines()]

        # sorting and fixing random seed for reproducibility
        manifest_data = sorted(manifest_data, key=lambda x: x['audio_filepath'])
        sample_idxs = list(range(len(manifest_data)))
        rng = np.random.RandomState(0)
        rng.shuffle(sample_idxs)

        duration = sum([x['duration'] for x in manifest_data])
        validation_duration, test_duration = 1800, 3600  # 30 minutes, 1 hour
        if duration <= 3600:  # 1 hour
            validation_duration, test_duration = 300, 600  # 5 minutes, 10 minutes
        elif duration > 3600 and duration <= 9000:  # 2.5 hours
            validation_duration, test_duration = 900, 1800  # 15 minutes, 30 minutes

        split_data = {}
        split_data['dev'] = self._accumulate_samples(manifest_data, sample_idxs, validation_duration)
        split_data['test'] = self._accumulate_samples(manifest_data, sample_idxs, test_duration)
        split_data['train'] = (
            [manifest_data[x] for x in sample_idxs],
            sum([manifest_data[x]['duration'] for x in sample_idxs]),
        )

        for split in ['train', 'dev', 'test']:
            actual_stats = (len(split_data[split][0]), round(split_data[split][1], 3))
            if EXPECTED_SPLIT_STATS[(self.dialect, split)] != actual_stats:
                raise RuntimeError(
                    f"Generated split stats (num files, duration) = {actual_stats}. "
                    f"But expected to see {EXPECTED_SPLIT_STATS[(self.dialect, split)]}. "
                    f"Did you add some custom pre-processing that changes number of files or duration?"
                )

        number_of_entries = 0
        total_duration = 0
        os.makedirs(os.path.dirname(self.output_manifest_file), exist_ok=True)
        with open(self.output_manifest_file, "wt", encoding="utf8") as fout:
            for data_entry in tqdm(split_data[self.data_split][0]):
                json.dump(data_entry, fout)
                number_of_entries += 1
                total_duration += data_entry["duration"]
                fout.write("\n")

        logger.info("Total number of entries after processing: %d", number_of_entries)
        logger.info("Total audio duration (hours) after processing: %.2f", total_duration / 3600)

    def _accumulate_samples(
        self, manifest_data: List[dict], sample_idxs: List[int], duration_threshold: int
    ) -> Tuple[List[dict], float]:
        """Create a subset of the manifest data having duration less than duration_threshold.

        Args:
            manifest_data: data for the manifest file
            sample_idxs: list of available indices to pick a sample from the manifest data
            duration_threshold: maximum duration of the samples to be included in the subset

        Returns:
            tuple: The accumulated subset of the manifest data and total accumulated duration
        """
        accumulated_data = []
        accumulated_duration = 0
        while accumulated_duration <= duration_threshold:
            sample_idx = sample_idxs.pop(0)
            accumulated_data.append(manifest_data[sample_idx])
            accumulated_duration += manifest_data[sample_idx]['duration']
        return accumulated_data, accumulated_duration
