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

import argparse
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preparing MTEDX data")
    parser.add_argument("--extracted_data_path", required=True, help="Path to the downloaded and extracted data.")
    parser.add_argument(
        "--language_id",
        required=True,
        help="The name of the language, used to determine output file name mtedx_{language}.tgz",
    )
    parser.add_argument("--num_entries", default=2, type=int, help="How many flac files to be splitted")
    parser.add_argument("--test_data_folder", required=True, help="Where to place the prepared data")

    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        data_path = os.path.join(tmpdir_path, "data")
        os.makedirs(data_path, exist_ok=True)
        for split in ["train", "valid", "test"]:
            vtt_path_dest= os.path.join(data_path, split, "vtt")
            flac_path_dest= os.path.join(data_path, split,  "wav") 
            os.makedirs(vtt_path_dest, exist_ok=True)    
            os.makedirs(flac_path_dest, exist_ok=True)        
            for idx, vtt_file in enumerate(os.listdir(os.path.join(
                        args.extracted_data_path, "data", split, "vtt"))):
                if idx == args.num_entries:
                    break
                flac_file = vtt_file.split(".")[0] + ".flac"
                vtt_file_src = os.path.join(args.extracted_data_path,"data", split, "vtt", vtt_file)
                flac_file_src = os.path.join(args.extracted_data_path, "data", split, "wav", flac_file)
                shutil.copy(vtt_file_src, vtt_path_dest)
                shutil.copy(flac_file_src, flac_path_dest)
        with tarfile.open(os.path.join(args.test_data_folder, f"mtedx_{args.language_id}.tgz"), "w:gz") as tar:
            tar.add(tmpdir, arcname=f"mtedx_{args.language_id}")

