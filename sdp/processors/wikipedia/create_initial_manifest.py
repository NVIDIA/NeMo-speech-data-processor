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

import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from sdp.processors.base_processor import BaseProcessor
from sdp.utils.common import download_file, extract_archive


class CreateInitialManifestWiki(BaseProcessor):
    def __init__(
        self,
        lang: str,
        raw_data_dir: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lang = lang
        self.raw_data_dir = raw_data_dir

    def download_and_extract_file(self, url):
        """Download an URL to a file with a progress bar."""
        # Ensure the directory exists before downloading
        if not os.path.exists(self.raw_data_dir):
            os.makedirs(self.raw_data_dir)

        download_file(url, self.raw_data_dir)

        # Extract the filename from the URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        file_path = Path(self.raw_data_dir) / filename

        subprocess.run(["bzip2", "-dk", str(file_path)], check=True)
        print("Extraction completed")

    def run_wikiextractor(self, extracted_filename):
        """Run WikiExtractor on the extracted XML file."""
        output_path = os.path.join(self.raw_data_dir, 'extracted')
        os.makedirs(output_path, exist_ok=True)
        subprocess.run(
            ["wikiextractor", extracted_filename, "-b", "100M", "-o", output_path],  # Batch size of the output files.
            check=True,
        )
        print("WikiExtractor processing completed")

    def process(self):
        dump_url = f'https://dumps.wikimedia.org/{self.lang}wiki/latest/{self.lang}wiki-latest-pages-articles.xml.bz2'

        self.download_and_extract_file(dump_url)

        # Extract the filename without .bz2 extension
        parsed_url = urlparse(dump_url)
        filename = os.path.basename(parsed_url.path)
        extracted_filename = filename.rstrip('.bz2')

        extracted_full_path = Path(self.raw_data_dir) / extracted_filename
        print("Starting WikiExtractor...")
        self.run_wikiextractor(str(extracted_full_path))
