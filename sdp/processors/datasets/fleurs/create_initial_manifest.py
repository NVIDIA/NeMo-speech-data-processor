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

import fnmatch
import glob
import json
import os
import shutil
import typing
from urllib.parse import parse_qs, urlparse

from sdp.processors.base_processor import BaseProcessor, DataEntry
from sdp.utils.common import download_file, extract_archive


def get_fleurs_url_list(lang: str, split: list[str]) -> list[str]:
    urls = []
    # examples
    # "https://huggingface.co/datasets/google/fleurs/resolve/main/data/hy_am/audio/dev.tar.gz",
    # "https://huggingface.co/datasets/google/fleurs/resolve/main/data/hy_am/dev.tsv"

    base_url = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"

    base_lang_url = os.path.join(base_url, lang)
    tsv_url = f"{base_lang_url}/{split}.tsv"
    urls.append(tsv_url)

    tar_gz_url = f"{base_lang_url}/audio/{split}.tar.gz"
    urls.append(tar_gz_url)

    return urls


class CreateInitialManifestFleurs(BaseProcessor):
    """Processor to create initial manifest for the fleurs dataset.
    Dataset link: https://huggingface.co/datasets/google/fleurs
    Will download all files in parallel and create manifest file with the
    'audio_filepath' and 'text' fields
    Args:
        config (str): Which data set shoudld be processed
            - options are:
            TODO: Add all language options in the format
            "hy_am": armenian
            "ko_kr": korean
            ["all"] (for all datasets avalable)
        split (str): Which data split should be processed
            - options are:
            "test",
            "train",
            "dev"
        audio_dir (str): Path to folder where should the filed be donwloaded and extracted
    Returns:
       This processor generates an initial manifest file with the following fields::
            {
                "audio_filepath": <path to the audio file>,
                "text": <transcription>,
            }
    """

    def __init__(
        self,
        lang: str,
        split: str,
        raw_data_dir: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lang = lang
        self.split = split
        self.raw_data_dir = raw_data_dir

    def process_transcript(self, file_path: str) -> list[dict[str, typing.Any]]:
        """
        Parse transcript TSV file and put it inside manyfest.
        Assumes the TSV file has two columns: file name and text.
        """

        entries = []
        root = os.path.dirname(file_path)

        with open(file_path, encoding="utf-8") as fin:
            for line in fin:
                # Split the line into filename text using the tab delimiter
                parts = line.strip().split('\t')
                if len(parts) < 2:  # Skip lines that don't have at least 2 parts
                    continue

                file_name, transcript_text = parts[1], parts[2]
                wav_file = os.path.join(root, file_name)

                entry = {"audio_filepath": os.path.abspath(wav_file), "text": transcript_text}
                entries.append(entry)

        return entries

    def process_data(self, data_folder: str, manifest_file: str) -> None:
        files = []
        entries = []

        for root, _, filenames in os.walk(data_folder):
            for filename in fnmatch.filter(filenames, "*.tsv"):
                files.append(os.path.join(root, filename))

        for file in files:
            result = self.process_transcript(file)
            entries.extend(result)

        with open(manifest_file, "w", encoding="utf-8") as fout:
            for m in entries:
                fout.write(json.dumps(m, ensure_ascii=False) + "\n")

    def download_extract_files(self, dst_folder: str) -> None:
        """downloading and extracting files"""

        os.makedirs(dst_folder, exist_ok=True)

        # downloading all files
        for file_url in get_fleurs_url_list(self.lang, self.split):
            download_file(file_url, str(dst_folder))
        for data_file in glob.glob(f'{dst_folder}/*.tar.gz'):
            extract_archive(str(data_file), str(dst_folder), force_extract=True)

        # Organizing files into their respective folders
        folder_mappings = {'test': '*test.tsv', 'train': '*train.tsv', 'dev': '*dev.tsv'}

        for folder, file_pattern in folder_mappings.items():
            target_folder = os.path.join(dst_folder, folder)

            if not os.path.exists(target_folder):
                continue

            all_files = os.listdir(dst_folder)

            matching_files = fnmatch.filter(all_files, file_pattern)

            for file_name in matching_files:
                file_path = os.path.join(dst_folder, file_name)
                dest_file_path = os.path.join(target_folder, file_name)
                shutil.move(file_path, dest_file_path)
                print(f'Moved {file_path} to {dest_file_path}')

    def process(self):
        self.download_extract_files(self.raw_data_dir)
        self.process_data(self.raw_data_dir, self.output_manifest_file)