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

"""Will take the downloaded .tsv file and audios directory and create a version with only X entries."""

import argparse
import os
import csv
import shutil
import tarfile
import tempfile
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preparing Fleurs test data")
    parser.add_argument("--extracted_tsv_path", required=True, help="Path to the downloaded .tsv file.")
    parser.add_argument("--extracted_audios_dir", required=True, help="Path to the downloaded and extracted audios directory.")
    parser.add_argument(
        "--archive_file_stem",
        required=True,
        help="What the stem (ie without the 'tar.gz' bit) of the new archive file should be",
    )
    parser.add_argument("--num_entries", default=20, type=int, help="How many entries to keep (in each split)")
    parser.add_argument("--test_data_folder", required=True, help="Where to place the prepared data")

    args = parser.parse_args()
    os.makedirs(args.test_data_folder, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        with open(args.extracted_tsv_path, "rt", encoding="utf8") as fin, \
            open(os.path.join(args.test_data_folder, args.archive_file_stem + '.tsv'), "wt", encoding="utf8") as fout:
                csv_reader = csv.reader(fin, delimiter='\t')        # creating CSV reader object
                csv_writer = csv.writer(fout, delimiter='\t')       # creating CSV reader object
                
                for idx, row in enumerate(csv_reader):
                    if idx == args.num_entries:
                        break
                    
                    src_audio_path = os.path.join(args.extracted_audios_dir, row[1])
                    dst_audio_path = os.path.join(tmpdir, row[1])
                    shutil.copy(src_audio_path, dst_audio_path)
                    
                    csv_writer.writerow(row)
                    
        with tarfile.open(os.path.join(args.test_data_folder, f"{args.archive_file_stem}.tar.gz"), "w:gz") as tar:
            # has to be the same as what's before .tar.gz
            tar.add(tmpdir, arcname=args.archive_file_stem)
