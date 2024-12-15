from typing import List, Dict
import json
import os
from tqdm import tqdm

from sdp.data_units.data_entry import DataEntry
from sdp.data_units.abc_unit import DataSource, DataSetter
from sdp.data_units.cache import CacheDir, CACHE_DIR


class Manifest(DataSource):
    def __init__(self, filepath: str = None, cache_dir: CacheDir = CACHE_DIR):
        self.write_mode = "w"
        self.encoding = 'utf8'
        self.file = None

        if not filepath:
            filepath = cache_dir.make_tmp_filepath()

        super().__init__(filepath)
        os.makedirs(os.path.dirname(self.source), exist_ok=True)
    
    def read_entry(self):
        if self.source is None:
            raise NotImplementedError("Override this method if the processor creates initial manifest")

        with open(file=self.source, mode='r', encoding = 'utf8') as file:
            for line in file:
                yield json.loads(line)
    
    def read_entries(self, in_memory_chunksize = None):
        manifest_chunk = []
        for idx, data_entry in enumerate(self.read_entry(), 1):
            if not in_memory_chunksize:
                yield data_entry
            else:
                manifest_chunk.append(data_entry)      
                if idx % in_memory_chunksize == 0:
                    yield manifest_chunk
                    manifest_chunk = []
        if manifest_chunk:
            yield manifest_chunk
        
    def write_entry(self, data_entry: DataEntry):
        if not self.file:
            self.file = open(file=self.source, mode="w", encoding=self.encoding)
            self.write_mode = "a"
        
        self.update_metrics(data_entry)
        if data_entry.data:
            json.dump(data_entry.data, self.file, ensure_ascii=False)
            self.file.write("\n")
    
    def write_entries(self, data_entries):
        for data_entry in tqdm(data_entries):
            self.write_entry(data_entry)
        
        self.close()
        
    def close(self):
        if self.file:
            self.file.close()
            self.file = None

class ManifestsSetter(DataSetter):
    def __init__(self, processors_cfgs: List[Dict]):
        super().__init__(processors_cfgs)
    
    def is_manifest_resolvable(self, processor_idx: int):
        processor_cfg = self.processors_cfgs[processor_idx]

        if "input_manifest_file" not in processor_cfg:
            if processor_idx == 0: ## ToDo
                return

            if not ("output" in self.processors_cfgs[processor_idx - 1] and
                   isinstance(self.processors_cfgs[processor_idx - 1]["output"], Manifest)):
                raise ValueError()
            
    def set_processor_manifests(self, processor_idx: int):
        self.is_manifest_resolvable(processor_idx)
        
        processor_cfg = self.processors_cfgs[processor_idx]

        if "input_manifest_file" in processor_cfg:
            input_manifest = Manifest(processor_cfg.pop("input_manifest_file"))
        else:
            #1 st processor
            if processor_idx == 0:
                input_manifest = None ##ToDo
            else:
                input_manifest = self.processors_cfgs[processor_idx - 1]["output"]
        
        processor_cfg["input"] = input_manifest
        
        if "output_manifest_file" in processor_cfg:
            output_manifest = Manifest(processor_cfg.pop("output_manifest_file"))
        else:
            output_manifest = Manifest()

        processor_cfg["output"] = output_manifest
        
        self.processors_cfgs[processor_idx] = processor_cfg
        print(processor_idx, processor_cfg)
        

