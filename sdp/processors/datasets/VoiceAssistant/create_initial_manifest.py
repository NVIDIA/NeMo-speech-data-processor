import os
import json
import tempfile
from tqdm import tqdm
from itertools import chain
from tqdm.contrib.concurrent import process_map
from typing import List, Dict

from sdp.processors.base_processor import BaseParallelProcessor, DataEntry
from sdp.logging import logger

class CreateInitialManifesVoiceAssistant(BaseParallelProcessor):
    HF_REPO = "VocalNet/VoiceAssistant-430K-vocalnet"
    METADATA_FILE = "VoiceAssistant-430K.json"
    
    def __init__(
        self,
        output_dir: str,
        allow_resume: bool = True,
        dataloader_workers: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.output_audio_dir = os.path.join(output_dir, "audio")
        self.output_tokens_dir = os.path.join(output_dir, "tokens")
        self.allow_resume = allow_resume
        self.dataloader_workers = dataloader_workers
        self.max_workers = os.cpu_count() if self.max_workers == -1 else min(self.max_workers, os.cpu_count())

    def prepare(self):
        from huggingface_hub import hf_hub_download

        os.makedirs(self.output_audio_dir, exist_ok=True)
        os.makedirs(self.output_tokens_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = hf_hub_download(
            repo_id=self.HF_REPO, 
            repo_type='dataset',
            filename=self.METADATA_FILE,
            cache_dir=temp_dir
        )

            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = {sample['id'] : sample for sample in json.load(f)}

    def process_dataset_entry(self, data_entry):
        audio_filepath = None
        if data_entry['audio_data'] is not None:
            audio_filepath = os.path.join(self.output_audio_dir, data_entry['speech_path'])
            with open(audio_filepath, 'wb') as f:
                f.write(data_entry['audio_data'])
        
        tokens_path = None
        if data_entry['tokens_data'] is not None:
            tokens_path = os.path.join(self.output_tokens_dir, data_entry['units_path'])
            with open(tokens_path, 'wb') as f:
                f.write(data_entry['tokens_data'])
        
        data_entry = self.metadata.get(data_entry['id'], {})
        data_entry['speech'] = audio_filepath
        data_entry['units'] = tokens_path
        
        return [DataEntry(data=data_entry)]

    def process(self):
        from datasets import load_dataset
        from datasets.iterable_dataset import IterableDataset
        from torchdata.stateful_dataloader import StatefulDataLoader

        self.prepare()
        os.makedirs(os.path.dirname(self.output_manifest_file), exist_ok=True)

        filemode = "wt"
        if self.allow_resume:
            filemode = "at"

        with open(self.output_manifest_file, filemode, encoding="utf8") as fout:
            iterable_dataset = load_dataset(self.HF_REPO, streaming=True)
            dataloader = StatefulDataLoader(iterable_dataset['train'], batch_size = self.chunksize, num_workers=self.dataloader_workers, collate_fn=lambda x: x)
            dataloader_state_file = os.path.join(os.path.dirname(self.output_manifest_file), f"dataloader_state.json")
            
            if self.allow_resume:
                if os.path.exists(dataloader_state_file):
                    logger.info(f'Found dataloader state. Continue data loading')
                    with open(dataloader_state_file, 'r') as f:
                        state_dict = json.load(f) 
                    dataloader.load_state_dict(state_dict)  
                else:
                    logger.warning(f'Dataloader state file not found. Loading data from start')

            for batch in dataloader:
                data = chain(
                    *process_map(
                        self.process_dataset_entry,
                        batch,
                        max_workers=self.max_workers,
                        chunksize=self.chunksize,
                    )
                )

                for data_entry in tqdm(data):
                    if data_entry.data is None:
                        continue
                    json.dump(data_entry.data, fout, ensure_ascii=False)
                    self.number_of_entries += 1
                    self.total_duration += data_entry.data.get("duration", 0)
                    fout.write("\n")
                
                fout.flush()
                with open(dataloader_state_file, 'w') as dataloader_state:
                    json.dump(dataloader.state_dict(), dataloader_state)
                    
        self.finalize(self.test_cases)
