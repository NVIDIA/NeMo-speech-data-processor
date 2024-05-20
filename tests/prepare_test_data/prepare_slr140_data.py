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
import json
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preparing SLR140 test data")
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

        for audio_dir in Path(args.extracted_data_path).glob('*'):
            if not audio_dir.is_dir():
                continue

            transcript_path = audio_dir / "train.json"
            audio_tmpdir = tmpdir_path / audio_dir.stem / audio_dir.stem
            audio_tmpdir.mkdir(exist_ok=True, parents=True)

            with open(transcript_path, "rt", encoding="utf-8-sig") as fin, open(
                audio_tmpdir / "train.json", "wt", encoding="utf-8-sig"
            ) as fout:
                sample = [json.loads(line) for line in fin.readlines()][0][: args.num_entries]

                for entry in sample:
                    utt_id = entry['wav'].split('/')[-1]
                    utt_dir = entry['wav'].split('/')[-2]

                    utt_tmp_dir = audio_tmpdir / utt_dir
                    utt_tmp_dir.mkdir(exist_ok=True)

                    src_wav_path = audio_dir / utt_dir / utt_id
                    tgt_wav_path = utt_tmp_dir / utt_id
                    shutil.copy(src_wav_path, tgt_wav_path)

                fout.write(str(sample).replace("'", '"'))

            shutil.make_archive((tmpdir_path / audio_dir.stem), 'zip', (tmpdir_path / audio_dir.stem))

            print(os.listdir(tmpdir_path))

            shutil.rmtree((tmpdir_path / audio_dir.stem))

            print(os.listdir(tmpdir_path))

        test_data_folder = Path(args.test_data_folder)
        test_data_folder.mkdir(exist_ok=True, parents=True)

        with tarfile.open(os.path.join(args.test_data_folder, f"{args.archive_file_stem}.tar.gz"), "w:gz") as tar:
            # has to be the same as what's before .tar.gz
            tar.add(tmpdir, arcname=args.archive_file_stem)
