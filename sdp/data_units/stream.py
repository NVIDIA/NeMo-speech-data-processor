from io import BytesIO
from typing import List
import pickle
from tqdm import tqdm

from sdp.data_units.data_entry import DataSource, DataEntry
from sdp.data_units.manifest import Manifest
class Stream(DataSource):
    def __init__(self):
        self.rw_amount = 0
        self.rw_limit = 1

        super().__init__(BytesIO())
    
    def rw_control(func):
        def wrapper(self, *args, **kwargs):
            self.source.seek(0)
            func(self, *args, **kwargs)
            if self.rw_amount >= self.rw_limit:
                self.reset()
            return wrapper

    @rw_control
    def read(self):
        return pickle.load(self.source)

    @rw_control
    def write(self, data: List[DataEntry]):
       for data_entry in tqdm(data):
           self._add_metrics(data_entry)
           pickle.dump(data_entry, self.source)
           self.source.flush()
    
    def reset(self):
        self.source.truncate(0)


class StreamsSetter:
    def __init__(self, processors_cfgs):
        self.processors_cfgs = processors_cfgs
        self.reference_prefix = "stream"
    
    def resolve_stream(self, reference: str):
        reference = reference.replace(self.reference_prefix, "")
        if reference == "init":
            return Stream()

        current_item = self.processors_cfgs
        key_chain = reference.split('.')
        for key in key_chain: 
            try: 
                key = int(key)
            except ValueError:
                continue
            
            current_item = current_item[key]
        
        if not isinstance(current_item, Stream):
            raise ValueError()
        
        return current_item
        
    def is_manifest_to_stream(self, processor_idx):
        processor_cfg = self.processors_cfgs[processor_idx]

        if processor_cfg['_target_'] == "sdp.processors.ManifestToStream":
            if "input_manifest_file" not in processor_cfg:
                if not ("output" in self.processors_cfgs[processor_idx - 1] and 
                        isinstance(self.processors_cfgs[processor_idx - 1]["output"], Manifest)):
                    raise ValueError()
            
            if "output_stream" in processor_cfg:
                if processor_cfg["output_stream"] != f"{self.reference_prefix}:init":
                    raise ValueError()
                
            return True
        else:
            return False
    
    def as_manifest_to_stream(self, processor_idx):
        processor_cfg = self.processors_cfgs[processor_idx]
        
        if "input_manifest_file" in processor_cfg:
            input_manifest = Manifest(processor_cfg.pop("input_manifest_file"))
        else:
            input_manifest = self.processors_cfgs[processor_idx - 1]['output']
        
        processor_cfg["input"] = input_manifest

        if "output_stream" in processor_cfg:
            output_stream = self.resolve_stream(processor_cfg.pop("output_stream"))
        else:
            output_stream = Stream()
        
        processor_cfg["output"] = output_stream
        return processor_cfg

    def is_stream_to_manifest(self, processor_idx):
        processor_cfg = self.processors_cfgs[processor_idx]

        if self.processors_cfgs[processor_idx]['_target_'] == "sdp.processors.StreamToManifest":
            if "input_stream" in processor_cfg:
                if processor_cfg["input_stream"] == f"{self.reference_prefix}:init":
                    raise ValueError()
            else:
                if not ("output" in self.processors_cfgs[processor_idx - 1] and 
                        isinstance(self.processors_cfgs[processor_idx - 1]["output"], Stream)):
                    raise ValueError()
                        
            return True
        else:
            return False
    
    def as_stream_to_manifest(self, processor_idx):
        processor_cfg = self.processors_cfgs[processor_idx]
        if "input_stream" in processor_cfg:
            input_stream = self.resolve_stream(processor_cfg.pop("input_stream"))
        else:
            input_stream = self.processors_cfgs[processor_idx - 1]["output"]
        
        processor_cfg["input"] = input_stream
        
        if "output_manifest_file" in processor_cfg:
            output_manifest = Manifest(processor_cfg.pop("output_manifest_file"))
        else:
            output_manifest = Manifest()

        processor_cfg["output"] = output_manifest
        return processor_cfg

    def traverse_processor(self, cfg):
        if isinstance(cfg, list):
            for i, item in enumerate(cfg):
                cfg[i] = self.traverse_processor(item)
        elif isinstance(cfg, dict):
            for key, value in cfg.items():
                cfg[key] = self.traverse_processor(value)
        elif isinstance(cfg, str) and cfg.startswith(self.reference_prefix):
            cfg = self.resolve_stream(cfg)
        
        return cfg

    def is_stream_resolvable(self, processor_idx):
        processor_cfg = self.processors_cfgs[processor_idx]

        if "input_stream" in processor_cfg:
            if not processor_cfg["input_stream"].startswith(self.reference_prefix):
                raise ValueError()
            
            if processor_cfg["input_stream"] == f"{self.reference_prefix}:init":
                raise ValueError()
        
        else:
            if not(hasattr(self.processors_cfgs[processor_idx - 1], "output") and 
                   isinstance(self.processors_cfgs[processor_idx - 1]["output"], Stream)
            ):
                raise ValueError()
            
        if "output_stream" in processor_cfg:
            if processor_cfg["output_stream"] != f"{self.reference_prefix}:init":
                raise ValueError()
        
        return True

    def set_processor_streams(self, processor_idx: int):
        if self.is_manifest_to_stream(processor_idx):
            processor_cfg = self.as_manifest_to_stream(processor_idx)
        
        elif self.is_stream_to_manifest(processor_idx):
            processor_cfg = self.as_stream_to_manifest(processor_idx)
        
        elif self.is_stream_resolvable(processor_idx):
            processor_cfg = self.processors_cfgs(processor_idx)
            processor_cfg = self.traverse_processor(processor_cfg)
            
            processor_cfg["input"] = processor_cfg.pop("input_stream")
            processor_cfg["output"] = processor_cfg.pop("input_stream", Stream())
            
        self.processors_cfgs[processor_idx] = processor_cfg
        return processor_cfg
    
    #raise ValueError("Expected a Stream object for 'input'")
    #Manifest() without path -> auto tmp