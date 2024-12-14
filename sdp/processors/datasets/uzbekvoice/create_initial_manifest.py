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

import glob
import json
import os
import typing
import gdown

from sdp.processors.base_processor import BaseProcessor
from sdp.utils.common import extract_archive
from sdp.logging import logger


class CreateInitialManifestUzbekvoice(BaseProcessor):
    """
    Processor to create initial manifest for the Uzbekvoice dataset.

    Will download all files, extract them, and create a manifest file with the
    "audio_filepath", "text" and "duration" fields.

    Args:    
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
        raw_data_dir: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = raw_data_dir

    def download_extract_files(self, dst_folder: str) -> None:
        """downloading and extracting files"""

        os.makedirs(dst_folder, exist_ok=True)

        # downloading all files
        # for big files google drive doesn't allow to try downlaoding them more than once
        # so, in case of receiveing gdown error we need to download them manually

        #check if clisp.zip and uzbekvoice-dataset.zip are already in dst_folder
        if os.path.exists(os.path.join(dst_folder, 'clips.zip')) and os.path.exists(os.path.join(dst_folder, 'uzbekvoice-dataset.zip')):
            print("Files already exist in the folder. Skipping download.")
        else:
            print(f"Downloading files from {self.URL}...")
            try:
                gdown.download_folder(self.URL, output=dst_folder)
            except Exception as e:
                print("Error occured while downloading files from google drive. Please download them manually.")
                print("URL: ", self.URL)
                print("Error: ", e)
        for file in glob.glob(os.path.join(dst_folder, '*.zip')):
            extract_archive(file, str(dst_folder), force_extract=True)
            print(f"Extracted {file}")


    def process_transcript(self, file_path: str) -> list[dict[str, typing.Any]]:
        """
        Parse transcript JSON file and put it inside manifest.
        """

        entries = []
        root = os.path.join(self.raw_data_dir, 'clips')
        number_of_entries = 0
        total_duration = 0
        # parse json file and collect audio file path, transcript and lenght in entries
        with open(file_path, encoding="utf-8") as fin:
            data = json.load(fin)
            for entry in data:
                audio_file = os.path.join(root, entry["client_id"], entry["original_sentence_id"] + '.mp3')
                transcript = entry["original_sentence"]
                utter_length = entry["clip_duration"]
                number_of_entries += 1
                entries.append(
                    {
                        "audio_filepath": os.path.abspath(audio_file), 
                        "text": transcript, 
                        "duration": utter_length
                    }
                )
            

            logger.info("Total number of entries after processing: %d", number_of_entries)
            logger.info("Total audio duration (hours) after processing: %.2f", total_duration / 3600)

        return entries

    def process_data(self, data_folder: str, manifest_file: str) -> None:
        entries = self.process_transcript(os.path.join(data_folder, "uzbekvoice-dataset", "voice_dataset.json"))

        with open(manifest_file, "w", encoding="utf-8") as fout:
            for m in entries:
                fout.write(json.dumps(m, ensure_ascii=False) + "\n")



    def process(self):
        self.download_extract_files(self.raw_data_dir)
        self.process_data(self.raw_data_dir, self.output_manifest_file)
