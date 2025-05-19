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
from huggingface_hub import hf_hub_download

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
    shard = dict(
        audio = dict(key = f"data/{args.lang_subset}/audio/{args.shard_id}.tar.gz"),
        duration = dict(key = f"data/{args.lang_subset}/duration/{args.shard_id}.txt"),
        text = dict(key = f"data/{args.lang_subset}/text/{args.shard_id}.json")
    )

    # Temporary directory for downloads
    with tempfile.TemporaryDirectory() as tmpdir:
        for datatype in shard:
            tmp_filepath = hf_hub_download(
                repo_id="espnet/yodas2",
                repo_type="dataset",
                filename = shard[datatype]['key'],
                local_dir=tmpdir,
                local_dir_use_symlinks = False,
            )

            assert os.path.exists(tmp_filepath), f"{datatype} file missing after download ({tmp_filepath} not found)."
            shard[datatype]['src_filepath'] = tmp_filepath

            dest_filepath = os.path.join(test_data_folder, shard[datatype]['key'])
            shard[datatype]['dest_filepath'] = dest_filepath
            os.makedirs(os.path.dirname(dest_filepath), exist_ok=True)

            if args.num_entries == -1:
                shutil.move(tmp_filepath, dest_filepath)

        if args.num_entries != -1:    
            yodas_ids = []
            with open(shard['duration']['src_filepath'], 'r') as fin, open(shard['duration']['dest_filepath'], 'w') as fout:
                for i, line in enumerate(fin, 1):
                    yodas_id = line.split()[0]
                    yodas_ids.append(yodas_id)
                    fout.write(line)
                    if i >= args.num_entries:
                        break
                else:
                    print(f"Warning: fewer lines than requested ({args.num_entries}).")
            
            with open(shard['text']['src_filepath'], 'r', encoding='utf8') as fin, open(shard['text']['dest_filepath'], 'w', encoding='utf8') as fout:
                all_samples = json.load(fin)
                selected = [s for s in all_samples if s['audio_id'] in yodas_ids]

                if len(selected) != len(yodas_ids):
                    raise ValueError("Mismatch between duration and text entries.")

                fout.write(json.dumps(selected) + '\n')
            
            # Extract and repack audio subset
            with tarfile.open(shard['audio']['src_filepath'], "r:gz") as tar_in, tarfile.open(shard['audio']['dest_filepath'], "w:gz") as tar_out:
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

        manifest_path = os.path.join(manifest_dir, "manifest_03.json")
        with open(manifest_path, 'w', encoding='utf8') as manifest:
            sample = {
                "lang_subset": args.lang_subset,
                "shard_id": args.shard_id,
                "audio_key": shard['audio']['key'],
                "duration_key": shard['duration']['key'],
                "text_key": shard['text']['key'],
                "src_lang": lang,
                "local_audio": shard['audio']['dest_filepath'],
                "local_duration": shard['duration']['dest_filepath'],
                "local_text": shard['text']['dest_filepath'],
            }
            manifest.write(json.dumps(sample) + '\n')