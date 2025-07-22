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

"""
Test module for the ConvertToTarredAudioDataset processor.

This test validates the correctness of the audio sharding logic under different configurations.
It generates a synthetic dataset of random WAV files with varying durations,
then checks if the processor correctly shards, buckets, and outputs the expected manifest entries.

Test optimization is achieved by using a pytest fixture to generate WAV files only once per test session.
"""

import os
import tempfile
import shutil
import wave
import numpy as np
import json
from typing import List, Dict
import pytest

from sdp.processors import ConvertToTarredAudioDataset

import os

if 'PATH' not in os.environ:
    os.environ['PATH'] = '/usr/bin:/bin'

EPSILON = 1e-7
NUM_WORKERS = max(1, os.cpu_count() // 2)

def generate_random_wav(audio_filepath: str, duration: float = 1.0, sample_rate: int = 16000):
    """Generate a mono 16-bit WAV file with random data of specified duration."""
    num_samples = int(duration * sample_rate)
    audio_data = np.random.randint(-32768, 32767, num_samples, dtype=np.int16)

    with wave.open(audio_filepath, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())


def generate_wav_subset(output_dir: str, min_duration: float = 0.1, max_duration: float = 70.0, num_samples: int = 70, sr: int = 16000) -> List[Dict]:
    """Generate a list of WAV files with increasing durations and return their metadata."""
    step = (max_duration - min_duration) / num_samples
    durations = [min_duration + i * step for i in range(num_samples)]

    samples = []
    for i, duration in enumerate(durations):
        sample_id = f'audio_{i}'
        audio_filepath = os.path.join(output_dir, f'{sample_id}.wav')
        generate_random_wav(audio_filepath, duration, sr)
        samples.append({
            'sample_id': sample_id,
            'audio_filepath': audio_filepath,
            'duration': duration
        })
    return samples


def write_manifest(samples: List[Dict], output_manifest_filepath: str):
    """Write a list of samples to a JSON lines manifest file."""
    with open(output_manifest_filepath, 'w', encoding='utf8') as manifest:
        for sample in samples:
            manifest.write(json.dumps(sample) + '\n')


def read_manifest(manifest_filepath: str) -> List[Dict]:
    """Read a JSON lines manifest file and return the list of samples."""
    with open(manifest_filepath, 'r', encoding='utf8') as manifest:
        return [json.loads(line) for line in manifest]


def strip_fields(samples: List[Dict], exclude_keys: List[str] = ['audio_filepath', 'abs_audio_filepath']) -> List[Dict]:
    """Remove specified keys from sample dictionaries."""
    return [{k: v for k, v in d.items() if k not in exclude_keys} for d in samples]

def get_expected_result(
    samples: List[Dict],
    num_shards=8,
    buckets_num=1,
    min_duration=1.0,
    max_duration=40.0,
    **kwargs
) -> List[Dict]:
    """
    Generate expected manifest given the parameters, matching real processor behavior:
    - sorted by duration
    - filtered by min/max
    - split into buckets
    - distributed by filling shard 0 completely, then shard 1, etc.
    - discard leftover samples that don't fit evenly
    """
    result = []
    EPSILON = 1e-7

    def process_bucket(bucket_samples: List[Dict], bucket_idx: int) -> None:
        bucket_samples = sorted(bucket_samples, key=lambda s: s['duration'])
        total = len(bucket_samples)
        per_shard = total // num_shards
        usable = per_shard * num_shards
        trimmed = bucket_samples[:usable]

        shard_size = usable // num_shards
        for shard_id in range(num_shards):
            start = shard_id * shard_size
            end = start + shard_size
            for s in trimmed[start:end]:
                s = s.copy()
                if buckets_num > 1:
                    # Only add bucket_id when bucketing is enabled (buckets_num > 1)
                    s['bucket_id'] = bucket_idx
                s['shard_id'] = shard_id
                result.append(s)

    # Fast path when no bucketing is requested
    if buckets_num == 1:
        filtered_samples = [
            s for s in samples
            if min_duration <= s['duration'] < max_duration  # strict upper bound (<) to match processor logic
        ]
        process_bucket(filtered_samples, 0)
        return result

    step = (max_duration + EPSILON - min_duration) / buckets_num

    for i in range(buckets_num):
        bucket_min = min_duration + i * step
        bucket_max = bucket_min + step

        # Strict upper bound (<) for all but the last bucket.
        upper = bucket_max + (EPSILON if i == buckets_num - 1 else 0.0)
        bucket_samples = [
            s for s in samples
            if bucket_min <= s['duration'] < upper
        ]
        process_bucket(bucket_samples, i)

    return result

# 🔧 Pytest fixture that generates audio samples once per test session
@pytest.fixture(scope="session")
def prepared_samples():
    """
    Generate and cache a set of audio samples for all test runs.
    Files are created in a temporary directory and deleted after the session.
    """
    safe_dir = tempfile.mkdtemp()
    samples = generate_wav_subset(safe_dir, min_duration=0.1, max_duration=70.0, num_samples=70, sr=16000)

    yield samples

    # Cleanup after session ends
    shutil.rmtree(safe_dir)


# Configuration parameters to test different behaviors of the processor
test_configs = [
    dict(num_shards=8, min_duration=1.0, max_duration=40.0, workers=NUM_WORKERS),
    dict(num_shards=4, buckets_num=2, min_duration=1.0, max_duration=40.0, workers=NUM_WORKERS),
    dict(num_shards=8, buckets_num=1, min_duration=1.0, max_duration=40.0, only_manifests=True, workers=NUM_WORKERS),
]

@pytest.mark.parametrize("cfg", test_configs)
def test_convert_to_tarred_audio_dataset(prepared_samples, cfg):
    """
    Test ConvertToTarredAudioDataset with different sharding and bucketing configurations.
    Checks both the manifest contents and the existence of expected output files.
    """
    with tempfile.TemporaryDirectory() as output_dir:
        input_manifest = os.path.join(output_dir, 'input.json')
        output_manifest = os.path.join(output_dir, 'output.json')
        cfg['target_dir'] = os.path.join(output_dir, 'tarred_dataset')
        cfg['sort_in_shards'] = True

        # Write manifest
        write_manifest(prepared_samples, input_manifest)

        # Run processor
        processor = ConvertToTarredAudioDataset(
            input_manifest_file=input_manifest,
            output_manifest_file=output_manifest,
            **cfg
        )
        processor.process()

        # Compare output manifest with expected values
        output_samples = sorted(strip_fields(read_manifest(output_manifest)), key=lambda x: x['duration'])
        expected_samples = sorted(strip_fields(get_expected_result(prepared_samples, **cfg)), key=lambda x: x['duration'])
        assert output_samples == expected_samples

        # Check existence of tar and manifest files
        base_dir = cfg['target_dir']
        bucket_dirs = (
            [os.path.join(base_dir, f"bucket{i+1}") for i in range(cfg.get('buckets_num', 1))]
            if cfg.get('buckets_num', 1) > 1
            else [base_dir]
        )

        for b_dir in bucket_dirs:
            for shard in range(cfg['num_shards']):
                # Tar files should exist unless we run in `only_manifests` mode.
                if not cfg.get('only_manifests', False):
                    assert os.path.exists(os.path.join(b_dir, f'audio_{shard}.tar'))

                # Per-shard manifests are written in `sharded_manifests/manifest_{shard}.json`,
                # unless this feature is explicitly disabled via `no_shard_manifests`.
                if not cfg.get('no_shard_manifests', False):
                    assert os.path.exists(os.path.join(b_dir, 'sharded_manifests', f'manifest_{shard}.json'))
