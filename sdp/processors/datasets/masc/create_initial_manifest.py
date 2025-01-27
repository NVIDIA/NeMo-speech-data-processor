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
import logging
from pathlib import Path
import pandas as pd
from sox import Transformer

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.utils.common import extract_archive

class CreateInitialManifestMASC(BaseParallelProcessor):
    """
    Processor for creating initial manifest for Massive Arabic Speech Corpus (MASC). \n
    Dataset link: https://ieee-dataport.org/open-access/masc-massive-arabic-speech-corpus.
    Prior to calling processor download the tarred dataset and store it under `raw_dataset_dir/masc.tar.gz`.
    
    Creates manifest from samples in . `dataset_dir/subsets/data_split.csv`. All meta information is kept.

    Args:
        raw_data_dir (str): The root directory of the dataset.
        extract_archive_dir (str): Directory where the extracted data will be saved.
        resampled_audios_dir (str): Directory where the resampled audio will be saved.
        data_split (str): Dataset split type.
        already_extracted (bool): If True, we will not try to extract the raw data. Defaults to False.
        target_samplerate (int): Sample rate (Hz) to use for resampling. Defaults to 16000.
        target_nchannels (int): Number of channels to create during resampling process. Defaults to 1.
        output_manifest_sample_id_key (str): The field name to store sample ID. Defaults to "sample_id".
        output_manifest_vtt_filapath_key (str): The field name to store vtt file path. Defaults to "vtt_filepath".
        output_manifest_audio_filapath_key (str): The field name to store audio file path. Defaults to "audio_filepath".
        verbose (bool): Set to True for more detailed logging.
        **kwargs: Additional keyword arguments to be passed to the base class `BaseParallelProcessor`.

    Returns:
        This processor generates an initial manifest file with the following fields::

            {
                "sample_id": <sample ID>
                "audio_filepath": <path to the audio file>,
                "vtt_filepath": <path to the vtt file>,
                "category": <video category>,
                "video_duration": <video duration>,
                "channel_id": <video channel ID>,
                "country": <video country>,
                "dialect": <speaker dialect>,
                "gender": <speaker gender>,
                "transcript_duration": <transcript duration>,
            }
    """

    def __init__(
        self,
        raw_data_dir: str,
        data_split: str,
        extract_archive_dir: str,
        resampled_audios_dir: str,
        already_extracted: bool = False,
        target_samplerate: int = 16000,
        target_nchannels: int = 1,
        output_manifest_sample_id_key: str = "sample_id",
        output_manifest_vtt_filapath_key: str = "vtt_filepath",
        output_manifest_audio_filapath_key: str = "audio_filepath",
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_dataset_dir = Path(raw_data_dir)
        self.data_split = data_split
        
        # in original dataset there are no train, dev and test splits. These are added to support end-to-end tests.
        if self.data_split == "train":
            self.data_split = "clean_train"
        if self.data_split == "dev":
            self.data_split = "clean_dev"
        if self.data_split == "test":
            self.data_split = "clean_test"
        
        self.extract_archive_dir = extract_archive_dir
        self.resampled_audios_dir = Path(resampled_audios_dir)
        self.already_extracted = already_extracted
        
        self.target_samplerate = target_samplerate
        self.target_nchannels = target_nchannels

        self.output_manifest_sample_id_key = output_manifest_sample_id_key
        self.output_manifest_vtt_filapath_key = output_manifest_vtt_filapath_key
        self.output_manifest_audio_filapath_key = output_manifest_audio_filapath_key
        
        self.verbose = verbose

        data_split_values = ["train", "dev", "test", "clean_train", "clean_dev", "clean_test", "noisy_train", "noisy_dev", "noisy_test"]
        if self.data_split not in data_split_values:
            raise ValueError(f'Data split value must be from {data_split_values}. "{self.data_split}" was given.')

    def prepare(self):
        # Extracting data (unless already done).
        if not self.already_extracted:
            tar_gz_filepath = Path(str(self.raw_dataset_dir)) / "masc.tar.gz"
            if not tar_gz_filepath.exists:
                raise RuntimeError(
                    f"Did not find any file matching {tar_gz_filepath}. "
                    "For MASC dataset we cannot automatically download the data, so "
                    "make sure to get the data from https://ieee-dataport.org/open-access/masc-massive-arabic-speech-corpus"
                    "and put it in the 'raw_dataset_dir' folder."
                )

            self.dataset_dir = Path(extract_archive(tar_gz_filepath, self.extract_archive_dir))
        else:
            logging.info("Skipping dataset untarring...")
            self.dataset_dir = Path(self.extract_archive_dir) / "masc"
            
        self.vtts_dir = self.dataset_dir / "subtitles"
        self.audios_dir = self.dataset_dir / "audios"
        if self.data_split == "clean_train" or self.data_split == "noisy_train":
            self.csv_filepath = self.dataset_dir / "subsets" / f"{self.data_split}.csv"
        else:
            self.csv_filepath = self.dataset_dir / "subsets" / f"{self.data_split}_meta.csv"

        if not self.csv_filepath.exists():
            raise FileNotFoundError(f"{self.csv_filepath} not found.")

        if not self.vtts_dir.exists():
            raise FileNotFoundError(f"{self.vtts_dir} not found.")

        if not self.audios_dir.exists():
            raise FileNotFoundError(f"{self.audios_dir} not found.")
        
        os.makedirs(self.resampled_audios_dir, exist_ok=True)
        
    def read_manifest(self):
        csv = pd.read_csv(self.csv_filepath)
        return [row.to_dict() for _, row in csv.iterrows()]

    def process_dataset_entry(self, sample_data):
        sample_id = sample_data["video_id"]
        source_audio_filepath = self.audios_dir / f"{sample_id}.wav"
        target_audio_filepath = self.resampled_audios_dir / f"{sample_id}.wav"
        vtt_filepath = self.vtts_dir / f"{sample_id}.ar.vtt"
        
        # if source audio or vtt file do not exist skip
        if not (os.path.exists(source_audio_filepath) and os.path.exists(vtt_filepath)):
            return []
        
        # if target audio exists skip resampling
        if not os.path.exists(target_audio_filepath):
            tfm = Transformer()
            tfm.rate(samplerate=self.target_samplerate)
            tfm.channels(n_channels=self.target_nchannels)
            tfm.build(input_filepath=source_audio_filepath, output_filepath=target_audio_filepath)
        elif self.verbose:
            logging.info(f"{target_audio_filepath} already exists. Skipping resampling")

        sample_data.pop("video_id")
        sample_data.update(
            {
                self.output_manifest_sample_id_key: sample_id,
                self.output_manifest_vtt_filapath_key: str(vtt_filepath),
                self.output_manifest_audio_filapath_key: str(target_audio_filepath),
            }
        )
        
        return [DataEntry(data=sample_data)]
