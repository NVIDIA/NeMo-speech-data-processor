# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
from sdp.processors.base_processor import BaseProcessor

from .utils.inference import WhisperInferenceConfig, InferenceConfig, ModelConfig, DatasetConfig
from .utils.inference import main as process_whisper_inference

from omegaconf import OmegaConf

class FasterWhisperInference(BaseProcessor):
    def __init__(self, 
                 input_manifest_file: str, 
                 output_manifest_file: str,
                 model_size_or_path: str,
                 output_dir: str = None ,
                 skip_corrupted: bool = False, 
                 save_timestamps_separately: bool = True,
                 slice_by_offset: bool = False, 
                 **kwargs):
        super().__init__(
            input_manifest_file=input_manifest_file,
            output_manifest_file=output_manifest_file,
        )

        model_cfg = kwargs.pop('model', {})
        model_cfg['model_size_or_path'] = model_size_or_path
        model_cfg = OmegaConf.structured(ModelConfig(**model_cfg))

        inference_cfg = OmegaConf.structured(InferenceConfig(**kwargs.get('inference', {})))

        if not output_dir:
            output_dir = os.splitext(output_dir)[0]

        dataset_cfg = DatasetConfig(manifest_filepath = input_manifest_file, 
                                    output_dir = output_dir, 
                                    skip_corrupted = skip_corrupted,
                                    save_timestamps_separately = save_timestamps_separately,
                                    offset = slice_by_offset)
        
        language_detection_only = kwargs.pop('language_detection_only', False)

        self.config = WhisperInferenceConfig(model=model_cfg, dataset=dataset_cfg, inference=inference_cfg, 
                                             language_detection_only = language_detection_only)
    
    def process(self):
        output_manifest_file = process_whisper_inference(self.config)
        os.replace(output_manifest_file, self.output_manifest_file)



    