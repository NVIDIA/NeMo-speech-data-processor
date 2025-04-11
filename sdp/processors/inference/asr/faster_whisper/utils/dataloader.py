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

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .logger import logger

from typing import Union, List
import time
import json

class AudioDataset(Dataset):
    def __init__(self, manifest_filepath: str):
        start_time = time.time()
        self.samples = []
        
        with open(manifest_filepath, 'r', encoding = "utf8") as manifest:
            self.samples = [json.loads(line) for line in manifest.readlines()]
            self.samples = sorted(self.samples, key=lambda x: float(x['duration']))
        
        logger.info(f"Manifest {manifest_filepath} uploaded in {time.time() - start_time:.2f} seconds. {len(self.samples)} samples found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: Union[int, List[int]]):
        if isinstance(idx, int):
            return self.samples[idx]
        elif isinstance(idx, list):
            return [self.samples[i] for i in idx]
        else:
            raise TypeError("Index must be an int or a list of ints")


class WhisperDataloader(DataLoader):
    def __init__(self, manifest_filepath: str, batch_size: int, num_replicas: int =  None, rank: int = None, sampler = None, shuffle: bool = False, **kwargs):
        self.manifest_filepath = manifest_filepath
        dataset = AudioDataset(manifest_filepath)
        
        sampler = None
        if num_replicas is not None and rank is not None:
            sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

        kwargs['dataset'] = dataset
        kwargs['batch_size'] = batch_size
        kwargs['sampler'] = sampler
        kwargs['pin_memory'] = True
        super().__init__(**kwargs)

