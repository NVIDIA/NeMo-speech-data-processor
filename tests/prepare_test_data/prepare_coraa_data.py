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
import tempfile
from pathlib import Path
import zipfile
import subprocess  # For external commands (e.g., for rar)
import random
import csv
import glob 

def create_zip_archive(source_dir, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(source_dir, '..')))

def create_rar_archive(source_dir, output_path):
    parent_dir = os.path.dirname(source_dir)
    target_folder_name = os.path.basename(source_dir)
    command = ['rar', 'a', '-r', '-v20m', output_path, target_folder_name]
    subprocess.run(command, check=True, cwd=parent_dir)

def sample_and_copy_entries(transcript_path, tmpdir_path, num_entries, extracted_data_path, output_metadata_path):
    with open(transcript_path, "rt", encoding="utf8") as fin:
        reader = csv.reader(fin)
        header = next(reader)
        selected_rows = random.sample(list(reader), num_entries)

    with open(output_metadata_path, "wt", encoding="utf8", newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(header)  # Write the header
        for row in selected_rows:
            filepath = row[0]
            src_path = os.path.join(extracted_data_path, filepath)
            tgt_path = os.path.join(tmpdir_path, filepath)
            os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
            shutil.copy(src_path, tgt_path)
            writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preparing Coraa test data")
    parser.add_argument("--extracted_data_path", required=True, help="Path to the downloaded and extracted data.")
    parser.add_argument("--num_entries", default=200, type=int, help="Number of entries to keep (in each split)")
    parser.add_argument("--test_data_folder", required=True, help="Where to place the prepared data")

    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        for split in ["train", "dev", "test"]:
            transcript_path = Path(args.extracted_data_path) / f"metadata_{split}_final.csv"
            output_metadata_path = Path(args.test_data_folder) / f"metadata_{split}_final.csv"
            sample_and_copy_entries(transcript_path, tmpdir_path, args.num_entries, args.extracted_data_path, output_metadata_path)
            archive_path = os.path.join(args.test_data_folder, split)
            source_dir = os.path.join(tmpdir_path, split)
            if split in ['dev', 'test']:
                create_zip_archive(source_dir, f"{archive_path}.zip")
            elif split == 'train':
                train_folder = os.path.join(args.test_data_folder, "train_dividido")
                os.makedirs(train_folder, exist_ok=True)
                create_rar_archive(source_dir, archive_path)
                pattern = os.path.join(args.test_data_folder, 'train*.rar')
                for file_path in glob.glob(pattern):
                    shutil.move(file_path,train_folder)
