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
from typing import Optional
from copy import deepcopy
from tqdm import tqdm
import shutil

from sdp.processors.base_processor import BaseProcessor
from sdp.processors.manage_files.utils.convert_to_tarred_audio_dataset import create_tar_datasets


@dataclass
class ConvertToTarredAudioDatasetConfig:
    """
    Configuration class for ConvertToTarredAudioDataset.

    Attributes:
        max_duration (float): Maximum allowed duration for audio samples.
        min_duration (Optional[float]): Minimum allowed duration for audio samples.
        concat_manifest_paths (Optional[str]): Path to a manifest file containing multiple manifest paths to concatenate.
        target_dir (Optional[str]): Output directory to save tarred dataset.
        metadata_path (Optional[str]): Path to write metadata about the tarred dataset.
        num_shards (int): Number of shards to create. If -1, it will be determined automatically.
        shuffle (bool): Whether to shuffle the input manifest before processing.
        keep_files_together (bool): If True, all segments from the same source file are kept in the same shard.
        sort_in_shards (bool): If True, samples inside each shard will be sorted by duration.
        buckets_num (int): Number of duration-based buckets to split data into.
        dynamic_buckets_num (int): Number of dynamic buckets for load balancing.
        shuffle_seed (Optional[int]): Random seed used for shuffling.
        write_metadata (bool): Whether to write metadata JSON files during processing.
        no_shard_manifests (bool): If True, disables writing per-shard manifest files.
        force_codec (Optional[str]): Audio codec to use when re-encoding audio files.
        workers (int): Number of worker processes for parallel audio re-encoding.
        slice_with_offset (bool): If True, audio slices will use offset and duration fields.
        only_manifests (bool): If True, only manifests will be generated without audio re-encoding.
    """
    max_duration: float
    min_duration: Optional[float] = None
    concat_manifest_paths: Optional[str] = None
    target_dir: Optional[str] = None
    metadata_path: Optional[str] = None
    num_shards: int = -1
    shuffle: bool = False
    keep_files_together: bool = False
    sort_in_shards: bool = False
    buckets_num: int = 1
    dynamic_buckets_num: int = 30
    shuffle_seed: Optional[int] = None
    write_metadata: bool = False
    no_shard_manifests: bool = False
    force_codec: Optional[str] = None
    workers: int = 1
    slice_with_offset: bool = False
    only_manifests: bool = False


class ConvertToTarredAudioDataset(BaseProcessor):
    """
    A processor for converting audio manifests into tarred audio datasets.

    This processor optionally splits data into duration-based buckets, and calls the
    `create_tar_datasets` utility to convert and shard audio data into tar files,
    with accompanying manifest files.

    Args:
        output_manifest_file (str): Path to the final output manifest.
        input_manifest_file (str): Path to the input manifest to be tarred.
        **cfg_kwargs: Additional keyword arguments passed to the configuration dataclass.
    
    Returns:
        Writes a tarred and sharded audio dataset to disk.

        - The dataset consists of multiple `.tar` archives with audio files.
        - A final manifest (JSON lines format) is written to ``output_manifest_file``, 
          referencing each sample, its path inside the tar, and other metadata.
        - If ``buckets_num > 1``, each sample will include an additional ``bucket_id`` field.

    .. note::
        If `buckets_num > 1`, the input manifest is split into multiple duration buckets,
        and each bucket is processed independently. A `bucket_id` is added to each sample.

        You may need to install the extra dependencies of Lhotse and NeMo for this processor to work correctly:
        ``pip install lhotse "nemo-toolkit[common]"``  
        
    """

    def __init__(
        self,
        output_manifest_file: str,
        input_manifest_file: str = None,
        **cfg_kwargs,
    ):
        super().__init__(
            input_manifest_file=input_manifest_file,
            output_manifest_file=output_manifest_file
        )
        self.cfg = ConvertToTarredAudioDatasetConfig(**cfg_kwargs)

    def process(self):
        # If bucketing is enabled, divide the data based on duration ranges.
        if self.cfg.buckets_num > 1:
            with open(self.output_manifest_file, 'w', encoding='utf8') as fout:
                bucket_length = (self.cfg.max_duration - self.cfg.min_duration) / float(self.cfg.buckets_num)

                for i_bucket in range(self.cfg.buckets_num):
                    # Create a config for the current bucket
                    bucket_config = deepcopy(self.cfg)
                    bucket_config.min_duration = self.cfg.min_duration + i_bucket * bucket_length
                    bucket_config.max_duration = bucket_config.min_duration + bucket_length
                    if i_bucket == self.cfg.buckets_num - 1:
                        # Ensure final bucket includes edge cases
                        bucket_config.max_duration += 1e-5

                    bucket_config.target_dir = os.path.join(self.cfg.target_dir, f"bucket{i_bucket+1}")
                    
                    print(f"Creating bucket {i_bucket+1} with min_duration={bucket_config.min_duration} and max_duration={bucket_config.max_duration} ...")
                    print(f"Results are being saved at: {bucket_config.target_dir}.")

                    # Create tarred dataset for the current bucket
                    create_tar_datasets(
                        manifest_path=self.input_manifest_file,
                        **vars(bucket_config)
                    )

                    # Read and modify the output manifest from this bucket
                    bucket_manifest_path = os.path.join(bucket_config.target_dir, 'tarred_audio_manifest.json')
                    with open(bucket_manifest_path, 'r', encoding='utf8') as bin_f:
                        for line in tqdm(bin_f, desc="Writing output manifest.."):
                            entry = json.loads(line)
                            entry['bucket_id'] = i_bucket
                            line = json.dumps(entry)
                            fout.writelines(f'{line}\n')

                    print(f"Bucket {i_bucket+1} is created.")

        else:
            # No bucketing — create single tarred dataset
            create_tar_datasets(
                manifest_path=self.input_manifest_file,
                **vars(self.cfg)
            )

            # Copy the generated manifest to the target location
            tarred_audio_manifest = os.path.join(self.cfg.target_dir, 'tarred_audio_manifest.json')
            shutil.copy(tarred_audio_manifest, self.output_manifest_file)