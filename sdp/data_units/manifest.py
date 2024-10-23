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
        with open(self.source, self.write_mode, encoding = 'utf8') as fout:
            for data_entry in tqdm(data):
                self._add_metrics(data_entry)
                json.dump(data_entry.data, fout, ensure_ascii=False)
                fout.write("\n")
            
            self.write_mode = 'a'


def set_manifests(processors_cfgs: List[Dict], cfg: List[Dict], tmp_dir: str):
    processors_cfgs_to_init = []

    cfg = OmegaConf.to_container(cfg)
    
    # special check for the first processor.
    # In case user selected something that does not start from
    # manifest creation we will try to infer the input from previous
    # output file
    if processors_cfgs[0] is not cfg['processors'][0] and "input_manifest_file" not in processors_cfgs[0]:
        # locating starting processor
        for idx, processor in enumerate(cfg['processors']):
            if processor is processors_cfgs[0]:  # we don't do a copy, so can just check object ids
                if "output_manifest_file" in cfg['processors'][idx - 1]:
                    processors_cfgs[0]["input_manifest_file"] = Manifest(cfg['processors'][idx - 1]["output_manifest_file"])
                break
    
    for idx, _processor_cfg in enumerate(processors_cfgs):
            processor_cfg = OmegaConf.to_container(_processor_cfg)
            # we assume that each processor defines "output_manifest_file"
            # and "input_manifest_file" keys, which can be optional. In case they
            # are missing, we create tmp files here for them
            # (1) first use a temporary file for the "output_manifest_file" if it is unspecified
            if "output_manifest_file" in processor_cfg:
                processor_cfg["output"] = Manifest(processor_cfg["output_manifest_file"])
            else:
                tmp_file_path = os.path.join(tmp_dir, str(uuid4()))
                processor_cfg["output"] = Manifest(tmp_file_path)
            
            processor_cfg.pop("output_manifest_file")

            # (2) then link the current processor's output_manifest_file to the next processor's input_manifest_file
            # if it hasn't been specified (and if you are not on the last processor)
            if "input_manifest_file" in processor_cfg:
                print('A' * 100)
                processor_cfg["input"] = Manifest(processor_cfg["input_manifest_file"])
            else:
                if idx > 0:
                    processor_cfg["input"] = Manifest(processors_cfgs[idx - 1]["output"].filepath)
            
            processor_cfg.pop("input_manifest_file")
            processors_cfgs_to_init.append(processor_cfg)

    print("=" * 100)
    print(processors_cfgs_to_init)
    print("=" * 100)

    return processors_cfgs_to_init