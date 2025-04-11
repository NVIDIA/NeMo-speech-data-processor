# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import json
import os
from tqdm import tqdm
from dataclasses import asdict, is_dataclass

def serialize(obj):
    if is_dataclass(obj):
        return asdict(obj)
    elif isinstance(obj, list):
        return [serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    return obj

def join_rank_manifests(rank_manifests_filepaths: str):
    output_manifest_filepath = os.path.join(os.path.dirname(rank_manifests_filepaths[0]), 'predictions_all.json')
    with open(output_manifest_filepath, 'w', encoding='utf8') as output_manifest:
        for rank_manifest_filepath in tqdm(rank_manifests_filepaths):
            with open(rank_manifest_filepath, 'r', encoding='utf8') as rank_manifest:
                for line in rank_manifest:
                    output_manifest.writelines(line)
    
    return output_manifest_filepath