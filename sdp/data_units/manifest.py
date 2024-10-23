from typing import List, Dict
from uuid import uuid4
import json
import os
from tqdm import tqdm
from omegaconf import OmegaConf

from sdp.data_units.data_entry import DataEntry, DataSource

class Manifest(DataSource):
    def __init__(self, filepath: str):
        self.write_mode = "w"
        super().__init__(filepath)
    
    def read(self, in_memory_chunksize: int):
        manifest_chunk = []
        for idx, data_entry in enumerate(self._read_manifest(), 1):
            manifest_chunk.append(data_entry)
            if idx % in_memory_chunksize == 0:
                yield manifest_chunk
                manifest_chunk = []
        if manifest_chunk:
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


def set_manifests(processor_cfg, previous_output, tmp_dir):
    if "output_manifest_file" in processor_cfg:
        processor_cfg["output"] = Manifest(processor_cfg["output_manifest_file"])
    else:
        tmp_file_path = os.path.join(tmp_dir, str(uuid4()))
        processor_cfg["output"] = Manifest(tmp_file_path)
    
    processor_cfg.pop("output_manifest_file")

    # (2) then link the current processor's output_manifest_file to the next processor's input_manifest_file
    # if it hasn't been specified (and if you are not on the last processor)
    if "input_manifest_file" in processor_cfg:
        processor_cfg["input"] = Manifest(processor_cfg["input_manifest_file"])
    else:
        if type(previous_output) is Manifest:
            processor_cfg["input"] = previous_output
        else:
            return ValueError()
    
    processor_cfg.pop("input_manifest_file")
    return processor_cfg