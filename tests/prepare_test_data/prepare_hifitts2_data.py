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

"""Copies HiFiTTS-2 manifests and audio into a new directory with fewer entries."""

import argparse
import json
import os
from pathlib import Path
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preparing HiFiTTS-2 test data")
    parser.add_argument(
        "--workspace_folder", required=True, type=Path, help="Path to workspace where dataset was downloaded."
    )
    parser.add_argument(
        "--audio_folder", default="audio_22khz", type=Path, required=False, help="Name of root folder with audio."
    )
    parser.add_argument("--test_data_folder", required=True, type=Path, help="Where to place the prepared data")
    parser.add_argument(
        "--manifest_filename", default="manifest_22khz.json", type=str, required=False, help="Name of manifest manifest."
    )
    parser.add_argument(
        "--chapters_filename", default="chapters_22khz.json", type=str, required=False, help="Name of chapter manifest."
    )
    parser.add_argument(
        "--error_filename", default="errors_22khz.json", type=str, required=False, help="Name of chapter error manifest."
    )
    parser.add_argument("--num_entries", default=20, type=int, help="How many entries to keep from each manifest")

    args = parser.parse_args()

    files_to_copy = [args.manifest_filename, args.chapters_filename, args.error_filename]

    os.makedirs(args.test_data_folder, exist_ok=True)
    # Copy manifest files
    for filename in files_to_copy:
        input_path = args.workspace_folder / filename
        output_path = args.test_data_folder / filename
        with open(input_path, "r", encoding="utf-8") as input_f:
            with open(output_path, "w", encoding="utf-8") as output_f:
                for i, line in enumerate(input_f):
                    if i >= args.num_entries:
                        break
                    output_f.write(line)

    # Copy audio
    manifest_path = args.test_data_folder / args.manifest_filename
    input_audio_dir = args.workspace_folder / args.audio_folder
    output_audio_dir = args.test_data_folder / args.audio_folder
    with open(manifest_path, "r", encoding="utf-8") as input_f:
        for i, line in enumerate(input_f):
            if i >= args.num_entries:
                break
            row = json.loads(line)
            audio_filepath = row["audio_filepath"]
            input_path = input_audio_dir / audio_filepath
            output_path = output_audio_dir / audio_filepath
            output_path.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(src=input_path, dst=output_path)