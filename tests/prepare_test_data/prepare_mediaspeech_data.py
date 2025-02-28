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
import glob
import tarfile
import tempfile
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preparing Mediaspeech test data")
    parser.add_argument("--extracted_data_path", required=True, help="Path to the downloaded and extracted data.")
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
        
        audio_filepaths = glob.glob(f"{args.extracted_data_path}/*.flac")
        for idx, src_audio_filepath in enumerate(audio_filepaths):
            if idx == args.num_entries:
                break
            
            sample_id = os.path.basename(src_audio_filepath).split(".")[0]
            src_text_filepath = os.path.join(args.extracted_data_path, f"{sample_id}.txt")
            dst_text_filepath = os.path.join(tmpdir, f"{sample_id}.txt")
            dst_audio_filepath = os.path.join(tmpdir, f"{sample_id}.flac")
            
            shutil.copy(src_text_filepath, dst_text_filepath)
            shutil.copy(src_audio_filepath, dst_audio_filepath)
            
        with tarfile.open(os.path.join(args.test_data_folder, f"{args.archive_file_stem}.tar.gz"), "w:gz") as tar:
            # has to be the same as what's before .tar.gz
            tar.add(tmpdir, arcname=args.archive_file_stem)