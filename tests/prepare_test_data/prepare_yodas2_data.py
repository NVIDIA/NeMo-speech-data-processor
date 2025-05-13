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
Script to download and prepare a specific shard of the YODAS2 dataset for testing or development purposes.

Main features:
- Downloads audio, duration, and text files for a given language subset and shard from Hugging Face (`espnet/yodas2`).
- Uses `huggingface_hub.snapshot_download` if available; otherwise falls back to direct `wget` download.
- Supports trimming the dataset to only the first N entries via the `--num_entries` flag.
- Repackages selected audio entries into a new `.tar.gz`.
- Generates a minimal manifest JSON with references to the downloaded local files.
"""

import argparse
import os
import json
from pathlib import Path
import tempfile
import shutil
import tarfile
import io
import wget

try:
    from huggingface_hub import snapshot_download
    IS_HF_HUB_AVAILABLE = True
except ModuleNotFoundError:
    IS_HF_HUB_AVAILABLE = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a specific YODAS2 test data shard for local use.")

    parser.add_argument(
        "--lang_subset",
        default="en000",
        help=(
            "Language and subset ID from the YODAS2 dataset (e.g., 'en000', 'de003'). "
            "Used to construct the remote path: data/<lang_subset>/..."
        ),
    )

    parser.add_argument(
        "--shard_id",
        default="00000000",
        help=(
            "Shard ID to download (e.g., '00000000'). "
            "Used as a filename to identify the .tar.gz, .txt, and .json files."
        ),
    )

    parser.add_argument(
        "--num_entries",
        default=-1,
        type=int,
        help=(
            "If set to a positive integer, only the first N entries will be extracted "
            "from the shard (including audio, duration, and text). Use -1 to keep all data."
        ),
    )

    parser.add_argument(
        "--test_data_folder",
        required=True,
        help="Target directory where files and manifest will be saved.",
    )

    args = parser.parse_args()

    # Resolve target output directory and ensure it exists
    test_data_folder = Path(args.test_data_folder).resolve()
    os.makedirs(test_data_folder, exist_ok=True)

    # File keys used to construct Hugging Face paths
    audio_key = f"data/{args.lang_subset}/audio/{args.shard_id}.tar.gz"
    duration_key = f"data/{args.lang_subset}/duration/{args.shard_id}.txt"
    text_key = f"data/{args.lang_subset}/text/{args.shard_id}.json"

    # Local paths to store the downloaded files
    local_audio = os.path.join(test_data_folder, audio_key)
    local_duration = os.path.join(test_data_folder, duration_key)
    local_text = os.path.join(test_data_folder, text_key)

    # Ensure directories exist
    os.makedirs(os.path.dirname(local_audio), exist_ok=True)
    os.makedirs(os.path.dirname(local_duration), exist_ok=True)
    os.makedirs(os.path.dirname(local_text), exist_ok=True)

    # Temporary directory for downloads
    with tempfile.TemporaryDirectory() as tmpdir:
        downloaded_audio = os.path.join(tmpdir, audio_key)
        downloaded_duration = os.path.join(tmpdir, duration_key)
        downloaded_text = os.path.join(tmpdir, text_key)

        os.makedirs(os.path.dirname(downloaded_audio), exist_ok=True)
        os.makedirs(os.path.dirname(downloaded_duration), exist_ok=True)
        os.makedirs(os.path.dirname(downloaded_text), exist_ok=True)

        def download_with_hf():
            snapshot_download(
                repo_id="espnet/yodas2",
                repo_type="dataset",
                local_dir=tmpdir,
                allow_patterns=[audio_key, duration_key, text_key],
            )

        def download_with_wget():
            base_url = "https://huggingface.co/datasets/espnet/yodas2/resolve/main/"
            suffix = "?download=true"
            wget.download(f"{base_url}{audio_key}{suffix}", out=downloaded_audio)
            wget.download(f"{base_url}{duration_key}{suffix}", out=downloaded_duration)
            wget.download(f"{base_url}{text_key}{suffix}", out=downloaded_text)

        # Try downloading the files
        if IS_HF_HUB_AVAILABLE:
            try:
                download_with_hf()
            except Exception:
                download_with_wget()
        else:
            download_with_wget()

        # Sanity check to make sure files were downloaded
        assert os.path.exists(downloaded_audio), "Audio file missing after download."
        assert os.path.exists(downloaded_duration), "Duration file missing after download."
        assert os.path.exists(downloaded_text), "Text file missing after download."

        if args.num_entries == -1:
            # Move full shard
            shutil.move(downloaded_audio, local_audio)
            shutil.move(downloaded_duration, local_duration)
            shutil.move(downloaded_text, local_text)
        else:
            # Limit to N entries
            yodas_ids = []

            # Process duration file and extract first N IDs
            with open(downloaded_duration, 'r') as fin, open(local_duration, 'w') as fout:
                for i, line in enumerate(fin, 1):
                    yodas_id = line.split()[0]
                    yodas_ids.append(yodas_id)
                    fout.write(line)
                    if i >= args.num_entries:
                        break
                else:
                    print(f"Warning: fewer lines than requested ({args.num_entries}).")

            # Filter JSON entries by selected IDs
            with open(downloaded_text, 'r', encoding='utf8') as fin, open(local_text, 'w', encoding='utf8') as fout:
                all_samples = json.load(fin)
                selected = [s for s in all_samples if s['audio_id'] in yodas_ids]

                if len(selected) != len(yodas_ids):
                    raise ValueError("Mismatch between duration and text entries.")

                fout.write(json.dumps(selected) + '\n')

            # Extract and repack audio subset
            with tarfile.open(downloaded_audio, "r:gz") as tar_in, tarfile.open(local_audio, "w:gz") as tar_out:
                for yodas_id in yodas_ids:
                    filename = f"./{yodas_id}.wav"
                    member = tar_in.getmember(filename)
                    audio_bytes = tar_in.extractfile(member).read()
                    file_obj = io.BytesIO(audio_bytes)
                    member.size = len(audio_bytes)
                    tar_out.addfile(member, fileobj=file_obj)

    # Determine manifest location and folder
    lang = args.lang_subset[:2]
    manifest_dir = os.path.join(test_data_folder, lang)
    os.makedirs(manifest_dir, exist_ok=True)

    # Write output manifest
    manifest_path = os.path.join(manifest_dir, "manifest_03.json")
    with open(manifest_path, 'w', encoding='utf8') as manifest:
        sample = {
            "lang_subset": args.lang_subset,
            "shard_id": args.shard_id,
            "audio_key": audio_key,
            "duration_key": duration_key,
            "text_key": text_key,
            "src_lang": lang,
            "local_audio": local_audio,
            "local_duration": local_duration,
            "local_text": local_text,
        }
        manifest.write(json.dumps(sample) + '\n')