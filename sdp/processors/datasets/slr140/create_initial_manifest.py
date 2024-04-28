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
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import sox
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from sdp.logging import logger
from sdp.processors.base_processor import (
    BaseParallelProcessor,
    BaseProcessor,
    DataEntry,
)
from sdp.utils.common import download_file, extract_archive

DATASET_URL = "https://www.openslr.org/resources/140/{audio}.zip"

AVAILABLE_AUDIOS = [
    'audio2',
    'audio3',
    'audio4',
    'audio5',
]

SPLIT_STATS = {
    'test': 1 / 6,
    'dev': 1 / 12,
}


class CreateInitialManifestSLR140(BaseParallelProcessor):
    """Processor to create initial manifest for the SLR140 dataset.

    This is an open source Kazakh speech corpus developed by
    the Department of Artificial Intelligence and Big Data of Al-Farabi Kazakh National University.

    Args:
        raw_data_dir (str): where to put raw downloaded data.
        audios (list): should be the subset of the AVAILABLE_AUDIOS

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
        audios: Union[List[str], str],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = Path(raw_data_dir)
        self.audios = audios

        if self.audios == "all":
            self.audios = AVAILABLE_AUDIOS

        if any([audio not in AVAILABLE_AUDIOS for audio in self.audios]):
            raise ValueError(f"audios have to be one of {AVAILABLE_AUDIOS}")

    def prepare(self):
        """Downloading and extracting data (unless already done)."""
        os.makedirs(self.raw_data_dir, exist_ok=True)

        audio_urls = [DATASET_URL.format(audio=audio) for audio in self.audios]

        thread_map(
            download_file,
            audio_urls,
            [str(self.raw_data_dir)] * len(audio_urls),
            max_workers=self.max_workers,
            chunksize=self.chunksize,
        )

        for audio_url in audio_urls:
            extract_archive(str(self.raw_data_dir / os.path.basename(audio_url)), str(self.raw_data_dir))

        self.transcription_file = str(self.raw_data_dir / "{audio}" / "train.json")

    def read_manifest(self):
        if self.transcription_file is None:
            raise RuntimeError("self.process has to be called before processing the data.")

        dataset_entries = []

        for audio in self.audios:
            transcription_file = self.transcription_file.format(audio=audio)

            with open(transcription_file, "rt", encoding="utf-8-sig") as fin:
                audio_dataset_entries = [json.loads(line) for line in fin.readlines()][0]
                dataset_entries += audio_dataset_entries

        return dataset_entries

    def process_dataset_entry(self, data_entry: str):
        if len(data_entry) != 2:
            raise RuntimeError(f"Input data is badly formatted! Bad line: {data_entry}")

        audio_path = str(self.raw_data_dir / data_entry["wav"].replace("dataset/", ""))
        data = {
            "audio_filepath": audio_path,
            "duration": float(sox.file_info.duration(audio_path)),
            "text": data_entry["text"].strip(),
        }

        return [DataEntry(data=data)]


class CustomDataSplitSLR140(BaseProcessor):
    """Splits SLR140 data into train, dev or test subset.


    Args:
        data_split (str): "train", "dev" or "test".

    Returns:
        All the same fields as in the input manifest, but only a subset of
        the data is retained.
    """

    def __init__(self, data_split: str, split_audio_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.data_split = data_split
        self.split_audio_dir = split_audio_dir

    def process(self):
        with open(self.input_manifest_file, "rt", encoding="utf8") as fin:
            manifest_data = [json.loads(line) for line in fin.readlines()]

        # sorting and fixing random seed for reproducibility
        manifest_data = sorted(manifest_data, key=lambda x: x['audio_filepath'])
        sample_idxs = list(range(len(manifest_data)))
        rng = np.random.RandomState(0)
        rng.shuffle(sample_idxs)

        duration = sum([x['duration'] for x in manifest_data])
        validation_duration, test_duration = (duration * SPLIT_STATS['dev'], duration * SPLIT_STATS['test'])

        split_data = {}
        split_data['dev'] = self._accumulate_samples(manifest_data, sample_idxs, validation_duration)
        split_data['test'] = self._accumulate_samples(manifest_data, sample_idxs, test_duration)
        split_data['train'] = (
            [manifest_data[x] for x in sample_idxs],
            sum([manifest_data[x]['duration'] for x in sample_idxs]),
        )

        number_of_entries = 0
        total_duration = 0

        split_audio_dir = os.path.join(self.split_audio_dir, self.data_split)
        os.makedirs(split_audio_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.output_manifest_file), exist_ok=True)

        with open(self.output_manifest_file, "wt", encoding="utf8") as fout:
            for data_entry in tqdm(split_data[self.data_split][0]):
                audio_rel_path = os.path.relpath(
                    data_entry['audio_filepath'], os.path.join(self.split_audio_dir, "audios")
                )
                split_filepath = os.path.join(split_audio_dir, audio_rel_path)
                os.makedirs(os.path.dirname(split_filepath), exist_ok=True)
                os.rename(data_entry['audio_filepath'], split_filepath)
                data_entry['audio_filepath'] = split_filepath

                json.dump(data_entry, fout, ensure_ascii=False)
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
