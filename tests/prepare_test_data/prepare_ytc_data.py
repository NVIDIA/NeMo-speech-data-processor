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

"""Will take the downloaded tar file and create a version with only X entries."""

import argparse
import os
import shutil
import tarfile
import tempfile
import json
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preparing YTC test data")
    parser.add_argument("--extracted_data_path", required=True, help="Path to the downloaded and extracted data.")
    parser.add_argument(
        "--language",
        required=True,
        help="The name of the language, used to determine output file name ytc_{language}.tar.gz",
    )
    parser.add_argument("--num_entries", default=200, type=int, help="How many entries to keep (in each split)")
    parser.add_argument("--test_data_folder", required=True, help="Where to place the prepared data")

    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        split = "test"
        os.makedirs(tmpdir_path / split / "audio")
        manifest_path = tmpdir_path / split / "manifest.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as fout:
            for idx, audio_file in enumerate(Path(args.extracted_data_path).glob("audios/*")):
                if idx == args.num_entries:
                    break
                    
                # Copy audio file to temp directory maintaining relative path
                rel_path = audio_file.relative_to(Path(args.extracted_data_path))
                target_path = tmpdir_path / split / "audio" / rel_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(audio_file, target_path)
                stem = audio_file.stem
                
                # Write manifest entry
                manifest_entry = {
                    "audio_filepath": str(target_path.relative_to(tmpdir_path / split)),
                    "audio_item_id": stem
                }
                fout.write(f"{json.dumps(manifest_entry)}\n")
           
        os.makedirs(args.test_data_folder, exist_ok=True)
        with tarfile.open(os.path.join(args.test_data_folder, f"ytc_{args.language}.tar.gz"), "w:gz") as tar:
            # has to be the same as what's before .tar.gz
            tar.add(tmpdir, arcname=f"ytc_{args.language}")
