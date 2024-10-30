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

        if not filepath:
            filepath = cache_dir.make_tmp_filepath()

        super().__init__(filepath)
    
    def read(self, in_memory_chunksize: int = None):
        manifest_chunk = []
        for idx, data_entry in enumerate(self._read_manifest(), 1):
            if not in_memory_chunksize:
                yield data_entry
           
            manifest_chunk.append(data_entry)
            if in_memory_chunksize and idx % in_memory_chunksize == 0:
                yield manifest_chunk
                manifest_chunk = []
        if in_memory_chunksize and manifest_chunk:
            yield manifest_chunk

    def _read_manifest(self):
        if self.source is None:
            raise NotImplementedError("Override this method if the processor creates initial manifest")

        with open(self.source, 'r', encoding = 'utf8') as input_file:
            for line in input_file:
                yield json.loads(line)
    
    def write(self, data: List[DataEntry]):
        if self.write_mode == "w":
            os.makedirs(os.path.dirname(self.source), exist_ok=True)

        with open(self.source, self.write_mode, encoding = 'utf8') as fout:
            for data_entry in tqdm(data):
                self._add_metrics(data_entry)
                json.dump(data_entry.data, fout, ensure_ascii=False)
                fout.write("\n")
            
            self.write_mode = 'a'

class ManifestsSetter(DataSetter):
    def __init__(self, processors_cfgs: List[Dict]):
        super().__init__(processors_cfgs)
    
    def is_manifest_resolvable(self, processor_idx: int):
        processor_cfg = self.processors_cfgs[processor_idx]

        if "input_manifest_file" not in processor_cfg:
            if processor_idx == 0:
                pass

            if not("output" in self.processors_cfgs[processor_idx - 1] and
                   isinstance(self.processors_cfgs[processor_idx - 1]["output"]) is Manifest):
                raise ValueError()
            
    def set_processor_manifests(self, processor_idx: int):
        self.is_manifest_resolvable(processor_idx)
        
        processor_cfg = self.processors_cfgs[processor_idx]

        if "input_manifest_file" in processor_cfg:
            input_manifest = Manifest(processor_cfg.pop("input_manifest_file"))
        else:
            #1 st processor
            input_manifest = self.processors_cfgs[processor_idx - 1]["output"]
        
        processor_cfg["input"] = input_manifest
        
        if "output_manifest_file" in processor_cfg:
            output_manifest = Manifest(processor_cfg.pop("output_manifest_file"))
        else:
            output_manifest = Manifest()

        processor_cfg["output"] = output_manifest
        
        self.processors_cfgs[processor_idx] = processor_cfg
        return processor_cfg

