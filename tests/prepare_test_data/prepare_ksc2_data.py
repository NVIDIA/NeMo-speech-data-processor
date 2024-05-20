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

"""Will take the downloaded tar file and create a version with only X entries."""

import argparse
import csv
import json
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preparing KSC2 test data")
    parser.add_argument("--extracted_data_path", required=True, help="Path to the downloaded and extracted data.")
    parser.add_argument(
        "--archive_file_stem",
        required=True,
        help="What the stem (ie without the 'tar.gz' bit) of the new archive file should be",
    )
    parser.add_argument("--num_entries", default=200, type=int, help="How many entries to keep (in each audio)")
    parser.add_argument("--test_data_folder", required=True, help="Where to place the prepared data")

    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        split_dir = Path(args.extracted_data_path, "Train")
        tmp_split_dir = Path(tmpdir_path, "Train")
        tmp_split_dir.mkdir(exist_ok=True)

        for source_dir in split_dir.glob("*"):
            tmp_source_dir = Path(tmp_split_dir, source_dir.stem)
            tmp_source_dir.mkdir(exist_ok=True)

            for idx, audio_filepath in enumerate(source_dir.glob('*.flac')):
                if idx == args.num_entries:
                    break

                transcription_filepath = Path(audio_filepath.parent, audio_filepath.stem).with_suffix('.txt')

                if not transcription_filepath.exists():
                    transcription_filepath = transcription_filepath.with_suffix('.txt.txt')

                tgt_audio_path = Path(tmp_source_dir, audio_filepath.name)
                tgt_transcription_filepath = Path(tmp_source_dir, transcription_filepath.name)

                shutil.copy(audio_filepath, tgt_audio_path)
                shutil.copy(transcription_filepath, tgt_transcription_filepath)

        test_data_folder = Path(args.test_data_folder)
        test_data_folder.mkdir(exist_ok=True, parents=True)

        with tarfile.open(os.path.join(args.test_data_folder, f"{args.archive_file_stem}.tar.gz"), "w:gz") as tar:
            # has to be the same as what's before .tar.gz
            tar.add(tmpdir, arcname=args.archive_file_stem)
