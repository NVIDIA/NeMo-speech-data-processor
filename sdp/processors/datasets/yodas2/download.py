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

from pathlib import Path
import json
import os
from glob import glob
from tqdm import tqdm
import tempfile
import importlib.util

from huggingface_hub import hf_hub_download

from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor, BaseParallelProcessor
from sdp.processors.huggingface.huggingface_hub import ListRepoFiles, SnapshotDownload
from sdp.processors import ExtractTar

class ListYodas2Data(ListRepoFiles): 
    def __init__(self, use_metadata: bool = False, **kwargs):
        super().__init__(repo_id = "espnet/yodas2", repo_type = "dataset", **kwargs)
        self.use_metadata = use_metadata
    
    def process(self):
        if self.use_metadata:
            metadata = None
            with tempfile.TemporaryDirectory() as tmp_dir:
                yodas_metafile = hf_hub_download(repo_id="espnet/yodas2", filename="meta.py", repo_type="dataset", local_dir = tmp_dir)
                spec = importlib.util.spec_from_file_location("script", yodas_metafile)
                metadata = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata)
            
            with open(self.output_manifest_file, 'w', encoding='utf8') as fout:
                for lang_subset in sorted(metadata.lang2shard_cnt.keys()):
                    for shard_no in range(metadata.lang2shard_cnt['aa000']):
                        shard_id = str(shard_no).zfill(8)
                        data_entry = dict(lang_subset = lang_subset, shard_id = shard_id,
                                          audio_key = f'data/{lang_subset}/audio/{shard_id}.tar.gz',
                                          duration_key = f'data/{lang_subset}/duration/{shard_id}.txt',
                                          text_key = f'data/{lang_subset}/text/{shard_id}.json')
                        line = json.dumps(data_entry)
                        fout.writelines(f'{line}\n')
        else:       
            logger.info(f'Recieving files list of espnet/yodas2 dataset from Hugging Face..')
            self.list_repo_files()
            logger.info(f'Metadata have beeen successfully recieved. Aggregating filenames into shards..')

            lang2shard_files = {}
            
            for file in tqdm(self.files):
                if not file.startswith('data/'):
                    continue

                path = Path(file)
                lang_subset = path.parts[1]
                if lang_subset not in lang2shard_files:
                    lang2shard_files[lang_subset] = dict()
                lang_shards = lang2shard_files[lang_subset]

                shard_no = path.parts[3].split('.')[0]
                if shard_no not in lang_shards:
                    lang_shards[shard_no] = dict()
                shard_files = lang_shards[shard_no]

                data_type = path.parts[2]
                shard_files[data_type] = file

            logger.info(f'Writing data into manifest..')

            with open(self.output_manifest_file, 'w', encoding='utf8') as fout:
                for lang_subset in sorted(lang2shard_files.keys()):
                    lang_subset_shards = lang2shard_files[lang_subset]
                    for shard_id in sorted(lang_subset_shards.keys()):
                        data_entry = dict(
                            lang_subset = lang_subset,
                            shard_id = shard_id,
                            )
                        
                        shard_data = lang_subset_shards[shard_id]
                        for data_type in sorted(shard_data.keys()):
                            data_entry[f'{data_type}_key'] = shard_data[data_type]
                        
                        line = json.dumps(data_entry)
                        fout.writelines(f'{line}\n')
        
        logger.info(f'Metadata successfully saved!')
        

class DownloadYodas2Data(SnapshotDownload):
    def __init__(self, **kwargs):
        super().__init__(repo_id = "espnet/yodas2", repo_type = "dataset", **kwargs)
    
    def write_output_manifest_file(self): 
        samples = []
        with open(self.input_manifest_file, 'r', encoding = 'utf8') as fin, open(self.output_manifest_file, 'w', encoding = 'utf8') as fout:
            for line in fin:
                sample = json.loads(line)
                audio_file = sample.get('audio_key', None)
                if audio_file:
                    local_audio_file = os.path.join(self.local_dir, audio_file)
                    if os.path.exists(local_audio_file):
                        sample['local_audio'] = local_audio_file

                duration_file = sample.get('duration_key', None)
                if duration_file:
                    local_duration_file = os.path.join(self.local_dir, duration_file)
                    if os.path.exists(local_duration_file):
                        sample['local_duration'] = local_duration_file
                
                text_file = sample.get('text_key', None)
                if text_file:
                    local_text_file = os.path.join(self.local_dir, text_file)
                    if os.path.exists(local_text_file):
                        sample['local_text'] = local_text_file
                
                line = json.dumps(sample)
                fout.writelines(f'{line}\n')

    def process(self):
        allow_patterns = []
        with open(self.input_manifest_file, 'r', encoding = 'utf8') as fin:
            for line in fin:
                sample = json.loads(line)
                audio_file = sample.get('audio_key', None)
                if audio_file:
                    allow_patterns.append(audio_file)

                duration_file = sample.get('duration_key', None)
                if duration_file:
                    allow_patterns.append(duration_file)
                
                text_file = sample.get('text_key', None)
                if text_file:
                    allow_patterns.append(text_file)
    
        self.snapshot_download_kwargs['allow_patterns'] = allow_patterns
        self.download()
        self.write_output_manifest_file()


class ExtractYodas2Data(ExtractTar):
    def __init__(self, **kwargs):
        kwargs['get_extracted_filepaths'] = True
        super().__init__(**kwargs)
    
    def process_dataset_entry(self, data_entry):
        super().process_dataset_entry()
        audio_samples = []
        for audio_filepath in data_entry[self.output_filepath_field]:
            sample = dict(data_entry['lang_'])