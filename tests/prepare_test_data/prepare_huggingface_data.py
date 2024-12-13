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
import tempfile
import itertools
from pathlib import Path

if __name__ == "__main__":
    from datasets import load_dataset, Dataset, load_from_disk
    
    parser = argparse.ArgumentParser("Preparing TarteelAI's EveryAyah test data")
    parser.add_argument("--dataset_name", required=True, help="Hugging Face dataset name. E.g., 'tarteel-ai/everyayah'")
    parser.add_argument(
        "--archive_file_stem",
        required=True,
        help="What the stem (ie without the '.hf' bit) of the new archive file should be",
    )
    parser.add_argument("--data_split", default="test", help="Dataset data split")
    parser.add_argument("--num_entries", default=20, type=int, help="How many entries to keep (in each split)")
    parser.add_argument("--test_data_folder", required=True, help="Where to place the prepared data")

    args = parser.parse_args()
    
    os.makedirs(args.test_data_folder, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        dataset = load_dataset(args.dataset_name, split="train", streaming=True)
        sampled_dataset = list(itertools.islice(dataset, args.num_entries))
        sampled_dataset = Dataset.from_list(sampled_dataset)
        sampled_dataset.save_to_disk(os.path.join(args.test_data_folder, f"{args.archive_file_stem}.hf"))