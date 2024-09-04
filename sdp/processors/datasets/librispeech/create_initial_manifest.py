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

import fnmatch
import glob
import json
import os
import typing

from sdp.processors.base_processor import BaseProcessor
from sdp.utils.common import download_file, extract_archive


def get_librispeech_url_list(split: str) -> str:
    urls = {
        "dev-clean": "https://openslr.org/resources/12/dev-clean.tar.gz",
        "dev-other": "https://openslr.org/resources/12/dev-other.tar.gz",
        "test-clean": "https://openslr.org/resources/12/test-clean.tar.gz",
        "test-other": "https://openslr.org/resources/12/test-other.tar.gz",
        "train-clean-100": "https://openslr.org/resources/12/train-clean-100.tar.gz",
        "train-clean-360": "https://openslr.org/resources/12/train-clean-360.tar.gz",
        "train-other-500": "https://openslr.org/resources/12/train-other-500.tar.gz",
        "dev-clean-2": "https://www.openslr.org/resources/31/dev-clean-2.tar.gz",
        "train-clean-5": "https://www.openslr.org/resources/31/train-clean-5.tar.gz",
    }

    if split not in urls:
        valid_splits = ", ".join(urls.keys())
        raise ValueError(f"Invalid dataset split '{split}'. Valid options are: {valid_splits}")

    return urls[split]


class CreateInitialManifestLibrispeech(BaseProcessor):
    """Processor to create initial manifest for the Librispeech dataset.

    Dataset link: https://openslr.org/12
    Dataset link: https://openslr.org/31

    Will download all files, extract tars, and create a manifest file with the
    "audio_filepath" and "text" fields.

    Args:
        split (str): Which datasets or their combinations should be processed.
            Options are:

            - ``"dev-clean"``
            - ``"dev-other"``
            - ``"test-clean"``
            - ``"test-other"``
            - ``"train-clean-100"``
            - ``"train-clean-360"``
            - ``"train-other-500"``
            - ``"dev-clean-2"``
            - ``"train-clean-5"``

        raw_data_dir (str): Path to the folder where the data archive should be downloaded and extracted.

    Returns:
        This processor generates an initial manifest file with the following fields::

            {
                "audio_filepath": <path to the audio file>,
                "text": <transcription>,
            }
    """

    def __init__(
        self,
        split: str,
        raw_data_dir: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.split = split
        self.raw_data_dir = raw_data_dir

    def process_transcript(self, file_path: str) -> list[dict[str, typing.Any]]:
        """Parse transcript file and put it inside manifest
        We assume that flac files are located in the same directory as transcript file.
        """

        entries = []
        root = os.path.dirname(file_path)

        print(f"Processing transcript file: {file_path}") 
        with open(file_path, encoding="utf-8") as fin:
            for line in fin:
                id, text = line[: line.index(" ")], line[line.index(" ") + 1 :]
                transcript_text = text.strip()

                flac_file = os.path.join(root, id + ".flac")

                entry = {}
                entry["audio_filepath"] = os.path.abspath(flac_file)
                entry["text"] = transcript_text
                entries.append(entry)
        return entries

    def process_data(self, data_folder: str, manifest_file: str) -> None:
        split_folder = os.path.join(data_folder, "LibriSpeech", self.split)
        files = []
        entries = []
        if not os.path.exists(split_folder):
            raise FileNotFoundError(f"Directory for split '{self.split}' not found at {split_folder}")

        for root, _, filenames in os.walk(split_folder):
            for filename in fnmatch.filter(filenames, "*.trans.txt"):
                files.append(os.path.join(root, filename))

        for file in files:
            entries.extend(self.process_transcript(file))

        with open(manifest_file, "w") as fout:
            for entry in entries:
                fout.write(json.dumps(entry) + "\n")

    def download_extract_files(self, dst_folder: str) -> None:
        """downloading and extracting files"""

        os.makedirs(dst_folder, exist_ok=True)

        download_file(get_librispeech_url_list(self.split), str(dst_folder))
        data_file = f'{dst_folder}/{self.split}.tar.gz'
        extract_archive(str(data_file), str(dst_folder), force_extract=True)

    def process(self):
        self.download_extract_files(self.raw_data_dir)
        self.process_data(self.raw_data_dir, self.output_manifest_file)
