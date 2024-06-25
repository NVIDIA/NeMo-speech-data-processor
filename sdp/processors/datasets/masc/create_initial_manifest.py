# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import pandas as pd

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry


class CreateInitialManifestMASC(BaseParallelProcessor):
    """
    Processor for creating initial manifest for Massive Arabic Speech Corpus (MASC). \n
    Creates manifest from samples in .``dataset_dir/subsets/data_split.csv``. All meta information is kept.

    Args:
        dataset_dir (str): The root directory of the dataset.
        data_split (str): Dataset split type.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    Returns:
        This processor generates an initial manifest file with the following fields::

            {
                "sample_id": <sample ID>
                "audio_filepath": <path to the audio file>,
                "vtt_filepath": <path to the vtt file>,
                "category": <video category>,
                "video_duration": <video diration>,
                "channel_id": <video channel ID>,
                "country": <video country>,
                "dialect": <speaker dialect>,
                "gender": <speaker gender>,
                "transcript_duration": <transcript duration>,
            }
    """

    def __init__(
        self,
        dataset_dir: str,
        data_split: str,
        output_manifest_sample_id_key: str = "sample_id",
        output_manifest_vtt_filapath_key: str = "vtt_filepath",
        output_manifest_audio_filapath_key: str = "audio_filepath",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_dir = Path(dataset_dir)
        self.data_split = data_split

        self.output_manifest_sample_id_key = output_manifest_sample_id_key
        self.output_manifest_vtt_filapath_key = output_manifest_vtt_filapath_key
        self.output_manifest_audio_filapath_key = output_manifest_audio_filapath_key

        data_split_values = ["clean_train", "clean_dev", "clean_test", "noisy_train", "noisy_dev", "noisy_test"]
        if self.data_split not in data_split_values:
            raise ValueError(f'Data split value must be from {data_split_values}. "{self.data_split}" was given.')

        self.vtts_dir = os.path.join(dataset_dir, "subtitles")
        self.audios_dir = os.path.join(dataset_dir, "audios")
        if self.data_split == "clean_train" or self.data_split == "noisy_train":
            self.csv_filepath = os.path.join(dataset_dir, "subsets", f"{data_split}.csv")
        else:
            self.csv_filepath = os.path.join(dataset_dir, "subsets", f"{data_split}_meta.csv")

        if not os.path.exists(self.csv_filepath):
            raise FileNotFoundError(f"{self.csv_filepath} not found.")

        if not os.path.exists(self.vtts_dir):
            raise FileNotFoundError(f"{self.vtts_dir} not found.")

        if not os.path.exists(self.audios_dir):
            raise FileNotFoundError(f"{self.audios_dir} not found.")

    def read_manifest(self):
        csv = pd.read_csv(self.csv_filepath)
        return [row.to_dict() for _, row in csv.iterrows()]

    def process_dataset_entry(self, sample_data):
        sample_id = sample_data["video_id"]
        audio_filepath = os.path.join(self.audios_dir, sample_id + ".wav")
        vtt_filepath = os.path.join(self.vtts_dir, sample_id + ".ar.vtt")
        if not (os.path.exists(audio_filepath) and os.path.exists(vtt_filepath)):
            return []

        sample_data.pop("video_id")
        sample_data.update(
            {
                self.output_manifest_sample_id_key: sample_id,
                self.output_manifest_vtt_filapath_key: vtt_filepath,
                self.output_manifest_audio_filapath_key: audio_filepath,
            }
        )
        return [DataEntry(data=sample_data)]
