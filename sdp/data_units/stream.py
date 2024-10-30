from io import BytesIO
from typing import List
import pickle

from sdp.data_units.data_entry import DataEntry
from sdp.data_units.abc_unit import DataSource, DataSetter
from sdp.data_units.manifest import Manifest
class Stream(DataSource):
    def __init__(self):
        self.rw_amount = 0
        self.rw_limit = 1

        super().__init__(BytesIO())
    
    def rw_control(func):
        def wrapper(self, *args, **kwargs):
            self.source.seek(0)
            result = func(self, *args, **kwargs)
            self.rw_amount += 1
            if self.rw_amount >= self.rw_limit:
                self.reset()
            return result
        return wrapper

    @rw_control
    def read(self, *args, **kwargs):
        self.source.seek(0)
        data = [pickle.load(self.source)]
        return data

    @rw_control
    def write(self, data: List[DataEntry]):
        self.source.seek(0)
        data = list(data)
        for entry in data:
            self._add_metrics(entry)

        pickle.dump([entry.data for entry in data], self.source)
    
    def reset(self):
        self.source.truncate(0)


class StreamsSetter(DataSetter):
    def __init__(self, processors_cfgs):
        super().__init__(processors_cfgs)
        self.reference_stream_prefix = "stream"
            
    def resolve_stream(self, reference: str):
        reference = reference.replace(self.reference_stream_prefix + ":", "")
        if reference == "init":
            return Stream()

        current_item = self.processors_cfgs
        key_chain = reference.split('.')
        for key in key_chain: 
            if key.isdigit():
                key = int(key)
            
            
            #TODO: replace io_stream fields to "input" / "output"
            if isinstance(key, str) and key.endswith("_stream"):
                key = key.replace("_stream", "")

            current_item = current_item[key]
        
        if not isinstance(current_item, Stream):
            raise ValueError()
        
        current_item.rw_limit += 1
        return current_item
        
    def is_manifest_to_stream(self, processor_idx, dry_run: bool = False):
        processor_cfg = self.processors_cfgs[processor_idx]

        if processor_cfg['_target_'] == "sdp.processors.ManifestToStream":
            if "input_manifest_file" not in processor_cfg:
                if not ("output" in self.processors_cfgs[processor_idx - 1] and 
                        isinstance(self.processors_cfgs[processor_idx - 1]["output"], Manifest)):       
                    if dry_run:
                        return False
                    
                    raise ValueError()
            
            if "output_stream" in processor_cfg:
                if processor_cfg["output_stream"] != f"{self.reference_stream_prefix}:init":
                    if dry_run:
                        return False
                    
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

    def is_stream_to_manifest(self, processor_idx, dry_run: bool = False):
        processor_cfg = self.processors_cfgs[processor_idx]

        if self.processors_cfgs[processor_idx]['_target_'] == "sdp.processors.StreamToManifest":
            if "input_stream" in processor_cfg:
                if processor_cfg["input_stream"] == f"{self.reference_stream_prefix}:init":
                    if dry_run:
                        return False
                    
                    raise ValueError()
            else:
                if not ("output" in self.processors_cfgs[processor_idx - 1] and 
                        isinstance(self.processors_cfgs[processor_idx - 1]["output"], Stream)):
                    if dry_run:
                        return False
                    
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
            input_stream.rw_limit += 1
        
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
        elif isinstance(cfg, str) and cfg.startswith(self.reference_stream_prefix):
            cfg = self.resolve_stream(cfg)
        
        return cfg

    def is_stream_resolvable(self, processor_idx, dry_run: bool = False):
        processor_cfg = self.processors_cfgs[processor_idx]

        if "input_stream" in processor_cfg:
            if not processor_cfg["input_stream"].startswith(self.reference_stream_prefix):
                if dry_run:
                    return False

                raise ValueError()
            
            if processor_cfg["input_stream"] == f"{self.reference_stream_prefix}:init":
                if dry_run:
                    return False

                raise ValueError()
        
        else:
            if not(hasattr(self.processors_cfgs[processor_idx - 1], "output") and 
                   isinstance(self.processors_cfgs[processor_idx - 1]["output"], Stream)
            ):
                if dry_run:
                    return False
                
                raise ValueError()
            
        if "output_stream" in processor_cfg:
            if processor_cfg["output_stream"] != f"{self.reference_stream_prefix}:init":
                if dry_run:
                    return False
                
                raise ValueError()
        
        return True

    def set_processor_streams(self, processor_idx: int):
        if self.is_manifest_to_stream(processor_idx):
            processor_cfg = self.as_manifest_to_stream(processor_idx)
        
        elif self.is_stream_to_manifest(processor_idx):
            processor_cfg = self.as_stream_to_manifest(processor_idx)
        
        elif self.is_stream_resolvable(processor_idx):
            processor_cfg = self.processors_cfgs[processor_idx]
            processor_cfg = self.traverse_processor(processor_cfg)
            
            processor_cfg["input"] = processor_cfg.pop("input_stream")
            processor_cfg["output"] = processor_cfg.pop("output_stream", Stream())
            
        self.processors_cfgs[processor_idx] = processor_cfg
        #return processor_cfg
    
    #raise ValueError("Expected a Stream object for 'input'")
    #Manifest() without path -> auto tmp