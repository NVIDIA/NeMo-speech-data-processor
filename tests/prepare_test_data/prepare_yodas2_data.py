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
Script to download a specific shard of the YODAS2 dataset from Hugging Face and prepare test data.

This script:
- Downloads a specific language subset and shard from the Hugging Face `espnet/yodas2` dataset.
- Optionally truncates the dataset to a fixed number of entries (`--num_entries`).
- Extracts and repacks audio samples from a tarball.
- Writes a local manifest JSON with metadata pointing to the downloaded files.
"""

import argparse
import os
import json
from pathlib import Path
import tempfile
import shutil
import tarfile
import io
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
        "--num_entries",
        default=-1,
        type=int,
        help=(
            "If set to a positive number, limits the dataset to the first N entries. "
            "Only N lines from the duration file, corresponding samples from the text file, "
            "and matching audio files will be kept. Set to -1 to keep the full shard."
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
    local_audio = os.path.join(test_data_folder, audio_key)
    os.makedirs(os.path.dirname(local_audio), exist_ok=True)
    
    duration_key = f"data/{args.lang_subset}/duration/{args.shard_id}.txt"
    local_duration = os.path.join(test_data_folder, duration_key)
    os.makedirs(os.path.dirname(local_duration), exist_ok=True)
    
    text_key = f"data/{args.lang_subset}/text/{args.shard_id}.json"
    local_text = os.path.join(test_data_folder, text_key)
    os.makedirs(os.path.dirname(local_text), exist_ok=True)

    # Download the relevant files into a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = snapshot_download(
            repo_id="espnet/yodas2",
            repo_type="dataset",
            local_dir=tmpdir,
            allow_patterns=[audio_key, duration_key, text_key]
        )
        downloaded_duration = os.path.join(tmpdir, duration_key)
        downloaded_text = os.path.join(tmpdir, text_key)
        downloaded_audio = os.path.join(tmpdir, audio_key)

        # Sanity checks
        assert os.path.exists(downloaded_duration), "Missing duration file after download"
        assert os.path.exists(downloaded_text), "Missing text file after download"
        assert os.path.exists(downloaded_audio), "Missing audio file after download"

        # Full shard copy
        if args.num_entries == -1: 
            shutil.move(downloaded_duration, local_duration)
            shutil.move(downloaded_text, local_text)
            shutil.move(downloaded_audio, local_audio)

        # Partial shard (only first N entries)
        else:
            yodas_ids = []

            # Copy first N lines from duration file and collect audio IDs
            with open(downloaded_duration, 'r') as fin, open(local_duration, 'w') as fout:
                for i, line in enumerate(fin, 1):
                    yodas_id = line.split()[0]
                    yodas_ids.append(yodas_id)
                    fout.writelines(line)
                    if i >= args.num_entries:
                        break
                else:
                    print(f"Warning: available entries fewer than requested ({args.num_entries})")

            # Filter JSON entries by audio_id
            with open(downloaded_text, 'r', encoding='utf8') as fin, open(local_text, 'w', encoding='utf8') as fout:
                all_samples = json.load(fin)[0]
                selected = [sample for sample in all_samples if sample['audio_id'] in yodas_ids]

                if len(selected) != len(yodas_ids):
                    raise ValueError("Mismatch between duration IDs and filtered text samples.")

                fout.writelines(json.dumps(selected) + '\n')

            # Extract and re-tar only selected audio files
            with tarfile.open(downloaded_audio, "r:gz") as tar_in, tarfile.open(local_audio, "w:gz") as tar_out:
                for yodas_id in yodas_ids:
                    audio_filename = f'{yodas_id}.wav'
                    audio_bytes = tar_in.extractfile(audio_filename).read()
                    file_obj = io.BytesIO(audio_bytes)
                    info = tar_in.getmember(audio_filename)
                    info.size = len(audio_bytes)
                    tar_out.addfile(info, fileobj=file_obj)

    # Extract language code from lang_subset (e.g., 'en000' -> 'en')
    lang = args.lang_subset[:2]
    manifest_dir = os.path.join(test_data_folder, lang)
    os.makedirs(manifest_dir, exist_ok=True)

    # Write manifest with metadata for further testing
    manifest_path = os.path.join(manifest_dir, "manifest_03.json")
    with open(manifest_path, 'w', encoding='utf8') as manifest:
        sample = dict(
            lang_subset=args.lang_subset,
            shard_id=args.shard_id,
            audio_key=audio_key,
            duration_key=duration_key,
            text_key=text_key,
            src_lang=lang,
            local_audio=local_audio,
            local_duration=local_duration,
            local_text=local_text,
        )
        manifest.writelines(json.dumps(sample) + '\n')
