# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from omegaconf import OmegaConf
import hydra
import yaml
import traceback

from sdp.data_units.cache import CACHE_DIR
from sdp.data_units.manifest import ManifestsSetter
from sdp.data_units.stream import StreamsSetter

from sdp.logging import logger

# customizing logger
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '[SDP %(levelname)1.1s %(asctime)s %(module)s:%(lineno)d] %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
logger.handlers
logger.addHandler(handler)
logger.propagate = False

class SDPRunner(ManifestsSetter, StreamsSetter):
    def __init__(self, cfg: OmegaConf): 
        OmegaConf.resolve(cfg)
        self.processors_from_cfg = cfg.processors   
        self.processors_cfgs = self.select_processors_to_run(cfg.get("processors_to_run", "all"))
        self.processors = []

        self.use_streams = cfg.get("use_streams", False)
        
        super().__init__(self.processors_cfgs)
    
    def select_processors_to_run(self, processors_to_select: str):
        selected_cfgs = []
        if processors_to_select == "all":
            selected_cfgs = self.processors_from_cfg[:]
        elif ":" not in processors_to_select:
            selected_cfgs = [self.processors_from_cfg[int(processors_to_select)]]
        else:
            slice_obj = slice(*map(lambda x: int(x.strip()) if x.strip() else None, processors_to_select.split(":")))
            selected_cfgs = self.processors_from_cfg[slice_obj]
        
        processors_cfgs = []
        for processor_cfg in selected_cfgs:
            processor_cfg = OmegaConf.to_container(processor_cfg)
            should_run = processor_cfg.pop("should_run", True)
            if should_run:
                processors_cfgs.append(processor_cfg)

        return processors_cfgs
    
    def infer_init_input(self):
        if (self.processors_cfgs[0] is not self.processors_from_cfg[0] and
            "input_manifest_file" not in self.processors_cfgs[0]):

            for processor_idx, processor_cfg in enumerate(self.processors_from_cfg):
                if processor_cfg is self.processors_cfgs[0]:
                    if "output_manifest_file" in self.processors_from_cfg[processor_idx - 1]:
                        self.processors_cfgs[0]["input_manifest_file"] = self.processors_from_cfg[processor_idx - 1]["output_manifest_file"]
                    break

    def set(self):
        self.infer_init_input()

        for processor_idx in range(len(self.processors_cfgs)):
            if not self.use_streams:
                self.set_processor_manifests(processor_idx)
            
            else:
                if (self.is_manifest_to_stream(processor_idx, dry_run = True) or 
                    self.is_stream_to_manifest(processor_idx, dry_run = True) or
                    self.is_stream_resolvable(processor_idx, dry_run = True)):
                    self.set_processor_streams(processor_idx)
                else:
                    self.set_processor_manifests(processor_idx)
    
    def build_processors(self):
        for processor_cfg in self.processors_cfgs:
            processor = hydra.utils.instantiate(processor_cfg)
            self.processors.append(processor)
        
    
    def test_processors(self):
        for processor in self.processors:
            processor.test()
    
    def run(self):
        try:
            self.set()
            logger.info(
                "Specified to run the following processors:\n %s",
                (yaml.dump(self.processors_cfgs, default_flow_style=False)),
            )
            
            self.build_processors()
            self.test_processors()
            
            for processor in self.processors:
                logger.info('=> Running processor "%s"', processor)
                processor.process()
        
        except Exception:
            print(f"An error occurred: {traceback.format_exc()}")
        
        finally:
            CACHE_DIR.cleanup()