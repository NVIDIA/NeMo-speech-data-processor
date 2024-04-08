# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preparing MCV test data")
    parser.add_argument("--extracted_data_path", required=True, help="Path to the downloaded and extracted data.")
    parser.add_argument(
        "--archive_file_stem",
        required=True,
        help="What the stem (ie without the 'tar.gz' bit) of the new archive file should be",
    )
    parser.add_argument("--num_entries", default=200, type=int, help="How many entries to keep (in each split)")
    parser.add_argument("--test_data_folder", required=True, help="Where to place the prepared data")

    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        os.makedirs(tmpdir_path / "clips")
        for split in ["train", "dev", "test"]:
            transcript_path = Path(args.extracted_data_path) / f"{split}.tsv"
            with open(transcript_path, "rt", encoding="utf8") as fin, open(
                tmpdir_path / f"{split}.tsv", "wt", encoding="utf8"
            ) as fout:
                fout.write(fin.readline())  # just copy over header line
                for idx, line in enumerate(fin):
                    if idx == args.num_entries:
                        break
                    utt_id = line.split("\t")[1]
                    src_mp3_path = os.path.join(args.extracted_data_path, "clips", utt_id)
                    fout.write(line)
                    tgt_mp3_path = os.path.join(tmpdir_path, "clips", utt_id)
                    shutil.copy(src_mp3_path, tgt_mp3_path)

        os.makedirs(args.test_data_folder, exist_ok=True)
        with tarfile.open(os.path.join(args.test_data_folder, f"{args.archive_file_stem}.tar.gz"), "w:gz") as tar:
            # has to be the same as what's before .tar.gz
            tar.add(tmpdir, arcname=args.archive_file_stem)
