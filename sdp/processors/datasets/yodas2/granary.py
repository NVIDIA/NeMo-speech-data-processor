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
        super().__init__(repo_id="YODASEnj/YDS", repo_type="dataset", **kwargs)
        if lang not in self.AVAILABLE_LANGS:
            raise ValueError("")
        self.lang = lang

        self.translation = translation
        if self.lang == "en" and self.translation:
            logger.warning(f'There are no translations for `en` language.')
            self.translation = False
    
    def process(self):
        os.makedirs(os.path.dirname(self.output_manifest_file), exist_ok = True)
        with open(self.output_manifest_file, 'w', encoding='utf8') as fout:
            pattern = f"{self.lang}/{self.lang}*.json"
            if self.translation:
                pattern = f"Translation/{self.lang}_/{self.lang}*.jsonl"

            self.snapshot_download_kwargs['allow_patterns'] = pattern
            with tempfile.TemporaryDirectory() as tmp_dir: 
                self.snapshot_download_kwargs["local_dir"] = tmp_dir
                self.download()

                for manifest_filepath in sorted(glob(f"{tmp_dir}/{pattern}")):
                    with open(manifest_filepath, 'r', encoding='utf8') as fin:
                        for line in tqdm(fin, desc = f'Processing {os.path.basename(manifest_filepath)}'):
                            sample = json.loads(line)
                            new_sample = dict(source_lang = self.lang,
                                            target_lang = self.lang,
                                            yodas_id = sample['wav_id'],
                                            offset = sample['start_time'],
                                            duration = sample['duration'],
                                            text = sample['text'],
                                            answer = sample['text'],
                                            decodercontext = "",
                                            emotion = "<|emo:undefined|>",
                                            pnc = "pnc",
                                            itn = "itn",
                                            timestamp = "notimestamp", 
                                            diarize = "nodiarize")
                                                        
                            if self.translation:
                                new_sample['target_lang'] = "en"
                                new_sample['answer'] = sample['translation_en']
                                
                            fout.writelines(json.dumps(new_sample) + '\n')