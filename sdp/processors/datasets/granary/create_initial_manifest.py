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
from glob import glob
from tqdm import tqdm
import tempfile

from sdp.processors.huggingface.huggingface_hub import SnapshotDownload
from sdp.logging import logger

class GetGranarysYodas2(SnapshotDownload):
    AVAILABLE_LANGS = ["bg", "cs", "da", "de", "el",
                       "en", "es", "et", "fi", "fr",
                       "hr", "hu", "it", "lt", "lv",
                       "nl", "pl", "pt", "ro", "ru",
                       "sk", "sv", "uk"]

    def __init__(self, lang: str, translation: bool = False, **kwargs):
    
        if lang not in self.AVAILABLE_LANGS:
            raise ValueError("")
        self.lang = lang
        pattern = f"{self.lang}/{self.lang}*.json"

        self.translation = translation

        if self.translation:
            if self.lang == "en":
                logger.warning(f'There are no translations for `en` language.')
                self.translation = False
            else:
                 pattern = f"Translation/{self.lang}_/{self.lang}*.jsonl"

        if 'snapshot_download_args' not in kwargs:
            kwargs['snapshot_download_args'] = dict()

        kwargs['snapshot_download_args']['repo_id']="YODASEnj/YDS"
        kwargs['snapshot_download_args']['repo_type']="dataset"
        kwargs['snapshot_download_args']['allow_patterns']=pattern

        super().__init__(**kwargs)

    def process(self):
        with tempfile.TemporaryDirectory() as tmp_dir: 
            self.snapshot_download_args["local_dir"] = tmp_dir
            super().process()

            with open(self.output_manifest_file, 'w', encoding='utf8') as fout:
                for manifest_filepath in sorted(glob(f"{tmp_dir}/{self.snapshot_download_args['allow_patterns']}")):
                    with open(manifest_filepath, 'r', encoding='utf8') as fin:
                        for line in tqdm(fin, desc = f'Processing {os.path.basename(manifest_filepath)}'):
                            sample = json.loads(line)
                            new_sample = dict(source_lang = self.lang,
                                            target_lang = self.lang,
                                            yodas_id = sample['wav_id'],
                                            offset = sample['start_time'],
                                            duration = sample['duration'],
                                            text = sample['text'],
                                            answer = sample['text']
                                            )
                                                        
                            if self.translation:
                                new_sample['target_lang'] = "en"
                                new_sample['answer'] = sample['translation_en']
                                
                            fout.writelines(json.dumps(new_sample) + '\n')