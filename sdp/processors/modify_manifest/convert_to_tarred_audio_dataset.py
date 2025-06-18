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

import os
import json
from dataclasses import dataclass
from copy import deepcopy
from typing import Optional, List
import shutil
from tqdm import tqdm
from omegaconf import OmegaConf, MISSING

from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor

@dataclass
class ASRTarredDatasetConfig:
    num_shards: int = -1
    shuffle: bool = False
    max_duration: Optional[float] = MISSING
    min_duration: Optional[float] = MISSING
    shuffle_seed: Optional[int] = None
    sort_in_shards: bool = True
    shard_manifests: bool = True
    keep_files_together: bool = False
    force_codec: Optional[str] = None
    use_lhotse: bool = False
    use_bucketing: bool = False
    num_buckets: Optional[int] = 1
    dynamic_buckets_num: Optional[int] = 30
    bucket_duration_bins: Optional[list[float]] = None


class ConvertToTarredAudioDataset(BaseProcessor):
    """This processor converts an ASR dataset into a tarred format compatible with TarredAudioToTextDataLayer.

    It reads audio entries from a manifest file, optionally splits them into duration-based buckets,
    and creates tarball archives along with a corresponding manifest file. This is useful for efficient
    training on large datasets with NeMo or similar toolkits.

    Args:
        input_manifest_file (str): Path to the input manifest containing audio samples.
        output_manifest_file (str): Path to the output manifest file to be created.
        target_dir (str): Directory where tarballs and manifests will be saved.
        max_workers (int): Number of worker processes to use for data processing. Default: -1 (use all available cores).
        num_shards (int): Number of tarball shards to create. Default: -1 (determined automatically).
        shuffle (bool): Whether to shuffle samples before sharding. Default: False.
        max_duration (float): Maximum duration of audio samples to include. Required.
        min_duration (float): Minimum duration of audio samples to include. Optional.
        shuffle_seed (int, optional): Seed for reproducible shuffling. Default: None.
        sort_in_shards (bool): Whether to sort samples within shards by duration. Default: True.
        shard_manifests (bool): Whether to generate individual shard manifests. Default: True.
        keep_files_together (bool): If True, keeps samples from the same source file together. Default: False.
        force_codec (str, optional): Audio codec to use for re-encoding (e.g., 'flac', 'opus'). Default: None.
        use_lhotse (bool): Reserved for Lhotse support. Currently unused. Default: False.
        use_bucketing (bool): If True, enables bucketing logic. Default: False.
        num_buckets (int): Number of duration-based buckets to create. Default: 1 (no bucketing).
        dynamic_buckets_num (int): Used for estimating bucket duration bins. Default: 30.
        bucket_duration_bins (list of float, optional): Custom duration bin edges for bucketing. Default: None.

    Returns:
        A tarred audio dataset saved under `target_dir` with a manifest at `output_manifest_file`.
        If bucketing is used, bucket IDs will be added to each manifest entry.
    """

    def __init__(self,
                 input_manifest_file: str,
                 output_manifest_file: str,
                 target_dir: str,
                 max_workers: int = -1, 
                 **kwargs):
                
        super().__init__(input_manifest_file = input_manifest_file, 
                         output_manifest_file = output_manifest_file)
        
        self.target_dir = target_dir
        self.max_workers = max_workers
        self.config = OmegaConf.structured(ASRTarredDatasetConfig(**kwargs))
    
    def create_tar_datasets(self, min_duration: float, max_duration: float, target_dir: str):
        from sdp.utils.convert_to_tarred_audio_dataset import ASRTarredDatasetBuilder

        builder = ASRTarredDatasetBuilder()   
        logger.info("Creating new tarred dataset ...")
        config = OmegaConf.merge(
            deepcopy(self.config), dict(min_duration = min_duration, max_duration = max_duration)
            )
        builder.configure(config)
        builder.create_new_dataset(manifest_path=self.input_manifest_file, target_dir=target_dir, num_workers=self.max_workers)

    def process(self):
        output_bucket_dirs = []
        if self.config.num_buckets > 1:
            bucket_length = (self.config.max_duration - self.config.min_duration) / float(self.config.num_buckets)
            for i in range(self.config.num_buckets):
                min_duration = self.config.min_duration + i * bucket_length
                max_duration = min_duration + bucket_length
                if i == self.config.num_buckets - 1:
                    # add a small number to cover the samples with exactly duration of max_duration in the last bucket.
                    max_duration += 1e-5
                target_dir = os.path.join(self.target_dir, f"bucket{i+1}")
                output_bucket_dirs.append(target_dir)
                logger.info(f"Creating bucket {i+1} with min_duration={min_duration} and max_duration={max_duration} ...")
                logger.info(f"Results are being saved at: {target_dir}.")
                self.create_tar_datasets(min_duration=min_duration, max_duration=max_duration, target_dir=target_dir)
                logger.info(f"Bucket {i+1} is created.")
        else:
            self.create_tar_datasets(min_duration=self.config.min_duration, max_duration=self.config.max_duration, target_dir=self.target_dir)
            output_bucket_dirs.append(self.target_dir)

        self.finalize(output_bucket_dirs)

    def finalize(self, output_bucket_dirs: List[str]):
        logger.info(f'Creating output manifest file ({self.output_manifest_file})..')
        if len(output_bucket_dirs) == 1:
            shutil.copy(os.path.join(output_bucket_dirs[0], 'tarred_audio_manifest.json'), self.output_manifest_file) 
        else:
            with open(self.output_manifest_file, 'w', encoding = 'utf8') as fout:   
                for bucket_i, bucket_dir in tqdm(enumerate(output_bucket_dirs, 1)):
                    bucket_audio_manifest = os.path.join(bucket_dir, 'tarred_audio_manifest.json')
                    with open(bucket_audio_manifest, 'r', encoding = 'utf8') as fin:
                        for line in fin:
                            entry = json.loads(line)
                            entry['bucket_id'] = bucket_i
                            line = json.dumps(entry)
                            fout.writelines(f'{line}\n')
        
        logger.info(f'Output manifest file saved. Tarred audio dataset is created.')