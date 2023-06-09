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
    parser = argparse.ArgumentParser("Preparing VoxPopuli test data")
    parser.add_argument("--data_path", required=True, help="Path to the processed data.")
    parser.add_argument(
        "--language_id",
        required=True,
        help="The id of the language",
    )
    parser.add_argument("--num_entries", default=200, type=int, help="How many entries to keep (in each split)")
    parser.add_argument("--test_data_folder", required=True, help="Where to place the prepared data")

    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        os.makedirs(tmpdir_path / "transcribed_data" / args.language_id)

        for split in ["train", "dev", "test"]:
            transcript_path = Path(args.data_path) / "transcribed_data" / args.language_id / f"asr_{split}.tsv"
            with open(transcript_path, "rt", encoding="utf8") as fin, open(
                tmpdir_path / "transcribed_data" / args.language_id / f"asr_{split}.tsv", "wt", encoding="utf8"
            ) as fout:
                for idx, line in enumerate(fin):
                    if idx == args.num_entries + 1:
                        break
                    fout.write(line)
                    if idx == 0:  # skipping header
                        continue
                    utt_id, raw_text, norm_text, spk_id, _, gender, is_gold_transcript, accent = line.split("\t")
                    year = utt_id[:4]
                    src_audio_path = (
                        Path(args.data_path) / "transcribed_data" / args.language_id / year / (utt_id + ".ogg")
                    )
                    target_audio_dir = tmpdir_path / "transcribed_data" / args.language_id / year
                    os.makedirs(target_audio_dir, exist_ok=True)
                    shutil.copy(src_audio_path, target_audio_dir / (utt_id + ".ogg"))
            # even though the voxpopuli processor expects untarred folder,
            # we still tar it to save time on the download from s3
            with tarfile.open(os.path.join(args.test_data_folder, f"transcribed_data.tar.gz"), "w:gz") as tar:
                # has to be the same as what's before .tar.gz
                tar.add(tmpdir_path / "transcribed_data", arcname=f"transcribed_data")
