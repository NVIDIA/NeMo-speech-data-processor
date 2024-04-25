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


def get_librispeech_url_list(splits: list[str]) -> list[str]:
    urls = [
        "https://openslr.org/resources/12/dev-clean.tar.gz",
        "https://openslr.org/resources/12/dev-other.tar.gz",
        "https://openslr.org/resources/12/test-clean.tar.gz",
        "https://openslr.org/resources/12/test-other.tar.gz",
        "https://openslr.org/resources/12/train-clean-100.tar.gz",
        "https://openslr.org/resources/12/train-clean-360.tar.gz",
        "https://openslr.org/resources/12/train-other-500.tar.gz",
    ]
    if "all" not in splits:
        filtered_urls = [url for url in urls if url.split('/')[-1].split('.tar')[0] in splits]
    else:
        filtered_urls = urls

    if len(filtered_urls) == 0:
        raise ValueError("No data found")

    return filtered_urls


class CreateInitialManifestLibrispeech(BaseProcessor):
    """Processor to create initial manifest for the Librispeech dataset.

    Dataset link: https://openslr.org/12

    Will download all files, extract tars, and create a manifest file with the
    "audio_filepath" and "text" fields.

    Args:
        splits (list[str]): Which datasets or their combinations should be processed.
            Options are:

            - ``["dev-clean"]``
            - ``["dev-other"]``
            - ``["test-clean"]``
            - ``["test-other"]``
            - ``["train-clean-100"]``
            - ``["train-clean-360"]``
            - ``["train-other-500"]``
            - ``["all"]`` (for all datasets available)

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
        splits: list[str],
        raw_data_dir: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.splits = splits
        self.raw_data_dir = raw_data_dir

    def process_transcript(self, file_path: str) -> list[dict[str, typing.Any]]:
        """Parse transcript file and put it inside manifest
        We assume that flac files are located in the same directory as transcript file.
        """

        entries = []
        root = os.path.dirname(file_path)

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
        files = []
        entries = []

        for root, _, filenames in os.walk(data_folder):
            for filename in fnmatch.filter(filenames, "*.trans.txt"):
                files.append(os.path.join(root, filename))

        for file in files:
            result = self.process_transcript(file)
            entries.extend(result)

        with open(manifest_file, "w") as fout:
            for m in entries:
                fout.write(json.dumps(m) + "\n")

    def download_extract_files(self, dst_folder: str) -> None:
        """downloading and extracting files"""

        os.makedirs(dst_folder, exist_ok=True)

        # downloading all files
        for file_url in get_librispeech_url_list(self.splits):
            download_file(file_url, str(dst_folder))
        for data_file in glob.glob(f'{dst_folder}/*.tar.gz'):
            extract_archive(str(data_file), str(dst_folder), force_extract=True)

    def process(self):
        self.download_extract_files(self.raw_data_dir)
        self.process_data(self.raw_data_dir, self.output_manifest_file)
