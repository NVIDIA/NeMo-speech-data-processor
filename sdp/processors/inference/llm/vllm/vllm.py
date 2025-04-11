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

import yaml
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor

class vLLMInference(BaseProcessor):
    def __init__(self,
                 input_manifest_file: str, 
                 output_manifest_file: str,
                 prompt: str = None,
                 prompt_field: str = None,
                 prompt_file: str = None,
                 generation_field: str = 'generation',
                 **kwargs):
    
        super().__init__(
                        input_manifest_file=input_manifest_file,
                        output_manifest_file=output_manifest_file,
                        )
    
        self.prompt = prompt
        self.prompt_field = prompt_field
        self.generation_field = generation_field

        prompt_args_counter = sum([prompt is not None, prompt_field is not None, prompt_file is not None])
        if prompt_args_counter < 1:
            raise ValueError(f'One of `prompt`, `prompt_field` or prompt_file` should be provided.')
        elif prompt_args_counter > 1:
            err = []
            if prompt:
                err.append(f'`prompt` ({prompt})')
            if prompt_field:
                err.append(f'`prompt_field` ({prompt_field})')
            if prompt_file:
                err.append(f'`prompt_file` ({prompt_file})')

            err = ', '.join(err)
            raise ValueError(f'Found more than one prompt values: {err}.')
        
        if prompt_file:
            self.prompt = self._read_prompt_file(prompt_file)
        
        self.model_params = kwargs.get('model', {})
        self.sampling_params = SamplingParams(**kwargs.get('inference', {}))
        self.chat_template_params = kwargs.get('apply_chat_template', {})
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_params['model'])

    def _read_prompt_file(self, prompt_filepath):
        with open(prompt_filepath, 'r') as prompt: 
            return yaml.safe_load(prompt)
    
    def get_entry_prompt(self, data_entry):
        entry_chat = []
        prompt = self.prompt
        if self.prompt_field:
            prompt = data_entry[self.prompt_field]
        
        for role in prompt:
            entry_chat.append(dict(
                role=role,
                content=prompt[role].format(**data_entry)
            ))

        entry_prompt = self.tokenizer.apply_chat_template(
            entry_chat,
            **self.chat_template_params
        )

        return entry_prompt
    
    def process(self):
        entries = []
        entry_prompts = []
        with open(self.input_manifest_file, 'r', encoding='utf8') as fin:
            for line in tqdm(fin, desc = "Building prompts: "):
                data_entry = json.loads(line)
                entries.append(data_entry)
                entry_prompt = self.get_entry_prompt(data_entry)
                entry_prompts.append(entry_prompt)

        entry_prompts = entry_prompts

        llm = LLM(**self.model_params)
        outputs = llm.generate(entry_prompts, self.sampling_params)
    
        with open(self.output_manifest_file, 'w', encoding='utf8') as fout:
            for data_entry, output in tqdm(zip(entries, outputs)):
                data_entry[self.generation_field] = output.outputs[0].text
                line = json.dumps(data_entry)
                fout.writelines(f'{line}\n')



