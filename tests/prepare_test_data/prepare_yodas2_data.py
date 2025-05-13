# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Script to download a specific shard of the YODAS2 dataset from Hugging Face and prepare a local manifest
for testing purposes.

It downloads the specified audio, duration, and text files from the dataset repository, stores them
locally in a test directory, and generates a JSON manifest describing the file paths and metadata.
"""

import argparse
import os
import json
from pathlib import Path
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a specific YODAS2 test data shard for local use.")

    parser.add_argument(
        "--lang_subset",
        default="en000",
        help=(
            "Language and subset ID to use from the YODAS2 dataset (e.g., 'en000', 'de003'). "
            "This determines the subfolder within 'data/' on Hugging Face."
        ),
    )

    parser.add_argument(
        "--shard_id",
        default="00000000",
        help=(
            "ID of the specific shard to download (e.g., '00000000', '00000001'). "
            "Used to locate the appropriate audio, duration, and text files."
        ),
    )

    parser.add_argument(
        "--test_data_folder",
        required=True,
        help=(
            "Path to the local directory where the downloaded data and generated manifest should be saved."
        ),
    )

    args = parser.parse_args()

    # Resolve and create the output directory
    test_data_folder = Path(args.test_data_folder).resolve()
    os.makedirs(test_data_folder, exist_ok=True)

    # Construct relative dataset file keys to download
    audio_key = f"data/{args.lang_subset}/audio/{args.shard_id}.tar.gz"
    duration_key = f"data/{args.lang_subset}/duration/{args.shard_id}.txt"
    text_key = f"data/{args.lang_subset}/text/{args.shard_id}.json"

    # Download only the specified files from Hugging Face to the local directory
    test_data_folder = snapshot_download(
        repo_id="espnet/yodas",
        repo_type="dataset",
        local_dir=test_data_folder,
        allow_patterns=[audio_key, duration_key, text_key]
    )

    # Extract language code from lang_subset (e.g., 'en000' -> 'en')
    lang = args.lang_subset[:2]
    manifest_dir = os.path.join(test_data_folder, lang)
    os.makedirs(manifest_dir, exist_ok=True)

    # Generate a simple manifest file for testing
    manifest_path = os.path.join(manifest_dir, "manifest_03.json")
    with open(manifest_path, 'w', encoding='utf8') as manifest:
        sample = dict(
            lang_subset=args.lang_subset,
            shard_id=args.shard_id,
            audio_key=audio_key,
            duration_key=duration_key,
            text_key=text_key,
            src_lang=lang,
            local_audio=os.path.join(test_data_folder, audio_key),
            local_duration=os.path.join(test_data_folder, duration_key),
            local_text=os.path.join(test_data_folder, text_key),
        )
        # Write a single line JSON object representing the sample
        manifest.writelines(json.dumps(sample) + '\n')
