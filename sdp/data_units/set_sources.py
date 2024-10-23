from omegaconf import OmegaConf
from typing import Any, List, Dict, Optional, Iterable

from sdp.data_units.manifest import Manifest, set_manifests
from sdp.data_units.stream import Stream, set_streams


def set_sources(processors_cfgs: List[Dict], cfg: List[Dict], tmp_dir: str, use_streams: bool = False):
    processors_cfgs_to_run = []

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
    
    previous_output = None
    for idx, _processor_cfg in enumerate(processors_cfgs):
        processor_cfg = OmegaConf.to_container(_processor_cfg)

        if _processor_cfg["_target_"] == "":
            pass
        
        elif not use_streams:
            processor_cfg = set_manifests(processor_cfg, previous_output, tmp_dir)
        
        else:
            if ("output_manifest_file" in processor_cfg  or 
                "input_manifest_file" in processor_cfg):
                
                if ("input_stream" in processor_cfg or 
                    "output_stream" in processor_cfg):

                    raise ValueError()

                processor_cfg = set_manifests(processor_cfg, previous_output, tmp_dir)
            else:
                processor_cfg = set_streams(processor_cfg, previous_output)
        
        processors_cfgs_to_run.append(processor_cfg)
        previous_output = processor_cfg['output']

    return processors_cfgs_to_run