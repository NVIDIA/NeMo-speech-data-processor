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
import os
import shutil
import tarfile
import tempfile
import csv
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preparing MASC test data")
    parser.add_argument("--extracted_data_path", required=True, help="Path to the downloaded and extracted data.")
    parser.add_argument(
        "--archive_file_stem",
        required=True,
        help="What the stem (ie without the 'tar.gz' bit) of the new archive file should be",
    )
    parser.add_argument("--num_entries", default=10, type=int, help="How many entries to keep (in each split)")
    parser.add_argument("--test_data_folder", required=True, help="Where to place the prepared data")

    args = parser.parse_args()
    
    # Define a dictionary to map splits to filenames
    filename_map = {
        "train": "clean_train.csv",
        "dev": "clean_dev_meta.csv",
        "test": "clean_test_meta.csv"
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        os.makedirs(tmpdir_path / "audios")
        os.makedirs(tmpdir_path / "subtitles")
        os.makedirs(tmpdir_path / "subsets")
        
        for split in ["train", "dev", "test"]:
            transcript_path = Path(args.extracted_data_path) / "subsets" / filename_map[split]
            with open(transcript_path, "rt", encoding="utf8") as fin, open(tmpdir_path / "subsets" / filename_map[split], "wt", encoding="utf8") as fout:
                csv_reader = csv.reader(fin)        # creating CSV reader object
                csv_writer = csv.writer(fout)       # creating CSV reader object
                
                csv_writer.writerow(next(csv_reader))   # writing colomns line 
                for idx, row in enumerate(csv_reader):
                    if idx == args.num_entries:
                        break
                    utt_id = row[0]
                    
                    # copying audio file
                    src_audio_path = os.path.join(args.extracted_data_path, "audios", f"{utt_id}.wav")
                    tgt_audio_path = os.path.join(tmpdir_path, "audios", f"{utt_id}.wav")
                    shutil.copy(src_audio_path, tgt_audio_path)
                    
                    # copying transcription file
                    src_transcript_path = os.path.join(args.extracted_data_path, "subtitles", f"{utt_id}.ar.vtt")
                    tgt_transcript_path = os.path.join(tmpdir_path, "subtitles", f"{utt_id}.ar.vtt")
                    shutil.copy(src_transcript_path, tgt_transcript_path)
                    
                    csv_writer.writerow(row)

        os.makedirs(args.test_data_folder, exist_ok=True)
        with tarfile.open(os.path.join(args.test_data_folder, f"{args.archive_file_stem}.tar.gz"), "w:gz") as tar:
            # has to be the same as what's before .tar.gz
            tar.add(tmpdir, arcname=args.archive_file_stem)
