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

import concurrent.futures
import json
import os
import typing
from urllib.parse import parse_qs, urlparse

import requests

from sdp.processors.base_processor import BaseProcessor, DataEntry
from sdp.utils.common import download_file, extract_archive


def get_fleurs_url_list(config: str, split: str) -> list[str]:
    # URL to fetch JSON data
    json_url = "https://datasets-server.huggingface.co/splits?dataset=google%2Ffleurs"

    # Send a request to the URL and parse the JSON response
    response = requests.get(json_url)
    if response.status_code != 200:
        raise Exception("Failed to fetch data")

    data = response.json()

    # Base URL for constructing the download URLs
    base_url = "https://datasets-server.huggingface.co/first-rows?dataset=google%2Ffleurs"

    # Filter and construct the URLs
    filtered_urls = []
    for entry in data["splits"]:
        if (entry["config"] == config or config == 'all') and entry["split"] == split:
            download_url = f"{base_url}&config={config}&split={split}"
            filtered_urls.append(download_url)

    if len(filtered_urls) == 0:
        print(f"CONFIG: {config}\n SPLIT: {split}")
        raise ValueError("No data found for the specified config and split")

    return filtered_urls


def fetch_data(url: str) -> list[dict[str, typing.Any]]:
    try:
        # Fetching the data from the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an error if the request failed

        data = response.json()

        rows = data.get('rows', [])

        return rows

    except requests.RequestException as e:
        print(f"Error fetching data: {e}")


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
            "validation"
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
        config: str,
        split: str,
        audio_dir: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.split = split
        self.audio_dir = audio_dir

    def process_transcrip(self, url: str, data_folder: str) -> list[dict[str, typing.Any]]:
        entries = []

        data_rows = fetch_data(url)
        for row in data_rows:
            file_url = row['row']['audio'][0]['src']
            file_transcription = row['row']['transcription']
            file_name = '-'.join(file_url.split('/')[-5:])
            file_path = os.path.join(data_folder, file_name)
            entry = {}
            entry["audio_filepath"] = os.path.abspath(file_path)
            entry["text"] = file_transcription
            entries.append(entry)

        return entries

    def process_data(self, data_folder: str, manifest_file: str) -> None:
        entries = []

        urls = get_fleurs_url_list(self.config, self.split)
        for url in urls:
            result = self.process_transcrip(url, data_folder)
            entries.extend(result)

        with open(manifest_file, "w") as fout:
            for m in entries:
                fout.write(json.dumps(m) + "\n")

    def download_files(self, dst_folder: str) -> None:
        """Downloading files in parallel."""

        os.makedirs(dst_folder, exist_ok=True)
        tasks = []
        for url in get_fleurs_url_list(self.config, self.split):
            data_rows = fetch_data(url)
            for row in data_rows:
                file_url = row['row']['audio'][0]['src']
                file_name = '-'.join(file_url.split('/')[-5:])
                tasks.append((file_url, str(dst_folder), False, file_name))

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(download_file, *task) for task in tasks]

            # Wait for all futures to complete and handle exceptions
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error occurred: {e}")

    def process(self):
        self.download_files(self.audio_dir)
        self.process_data(self.audio_dir, self.output_manifest_file)
