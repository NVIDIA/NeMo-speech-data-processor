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
    parser = argparse.ArgumentParser("Preparing SLR102 test data")
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

        split_dir = Path(args.extracted_data_path, "Meta")
        tmp_split_dir = Path(tmpdir_path, "Meta")
        tmp_split_dir.mkdir(exist_ok=True)

        with open(Path(split_dir, "train.csv"), "rt", encoding="utf8") as csvfile_in, open(
            Path(tmp_split_dir, "train.csv"), "wt", encoding="utf8"
        ) as csvfile_out:
            reader = csv.DictReader(csvfile_in, delimiter=" ")
            headers = next(reader, None)  # skip the headers
            writer = csv.DictWriter(csvfile_out, fieldnames=headers, delimiter=" ")
            writer.writeheader()

            utt_used = []

            for idx, row in enumerate(reader):
                if idx == args.num_entries:
                    break
                writer.writerow(row)
                utt_used.append(row["uttID"])

        transcript_dir = Path(args.extracted_data_path, "Transcriptions")
        tmp_transcript_dir = Path(tmpdir_path, "Transcriptions")
        tmp_transcript_dir.mkdir(exist_ok=True)

        audios_dir = Path(args.extracted_data_path, "Audios_flac")
        tmp_audios_dir = Path(tmpdir_path, "Audios_flac")
        tmp_audios_dir.mkdir(exist_ok=True)

        for utt in utt_used:
            audio_path = Path(audios_dir, utt).with_suffix(".flac")
            transcript_path = Path(transcript_dir, utt).with_suffix(".txt")

            tgt_audio_path = Path(tmp_audios_dir, utt).with_suffix(".flac")
            tgt_transcript_path = Path(tmp_transcript_dir, utt).with_suffix(".txt")

            shutil.copy(audio_path, tgt_audio_path)
            shutil.copy(transcript_path, tgt_transcript_path)

        test_data_folder = Path(args.test_data_folder)
        test_data_folder.mkdir(exist_ok=True, parents=True)

        with tarfile.open(os.path.join(args.test_data_folder, f"{args.archive_file_stem}.tar.gz"), "w:gz") as tar:
            # has to be the same as what's before .tar.gz
            tar.add(tmpdir, arcname=args.archive_file_stem)
